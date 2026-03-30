import argparse
import queue
import struct
import sys
import threading
import time
import zlib
from dataclasses import dataclass, field

import adi
import cv2
import numpy as np


def ts() -> str:
    now = time.time()
    return time.strftime("%H:%M:%S.", time.localtime(now)) + f"{int((now % 1) * 1000):03d}"


FREQ = 5_000_000_000
SAMP_RATE = 1_000_000
SPS = 4
FREQ_OFFSET = 50_000
RX_BUFFER_SIZE = 4096
RX_THRESHOLD = 2000
MAX_PAYLOAD_BYTES = 96
TX_GAIN = -10
AMPLITUDE = 2 ** 14

DEFAULT_VIDEO = "test.mp4"
DEFAULT_WIDTH = 160
DEFAULT_HEIGHT = 90
DEFAULT_JPEG_QUALITY = 18
DEFAULT_TARGET_FPS = 1.5
DEFAULT_CHUNK_REPEATS = 3
DEFAULT_CHUNK_GAP = 0.004
DEFAULT_START_REPEATS = 4
DEFAULT_RX_FRAME_TIMEOUT = 2.5
DEFAULT_MAX_RX_FRAMES = 8
DEFAULT_TX_QUEUE = 4
DEFAULT_FRAME_PASSES = 3

PKT_START = 1
PKT_CHUNK = 2
PKT_END = 3

BARKER = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.int8)
BARKER_UPSAMPLED = np.repeat(BARKER, SPS).astype(np.complex64)


@dataclass
class StreamConfig:
    sdr_ip: str
    freq: int = FREQ
    sample_rate: int = SAMP_RATE
    sps: int = SPS
    freq_offset: int = FREQ_OFFSET
    rx_buffer_size: int = RX_BUFFER_SIZE
    rx_threshold: int = RX_THRESHOLD
    max_payload_bytes: int = MAX_PAYLOAD_BYTES
    tx_gain: int = TX_GAIN
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    target_fps: float = DEFAULT_TARGET_FPS
    chunk_repeats: int = DEFAULT_CHUNK_REPEATS
    chunk_gap: float = DEFAULT_CHUNK_GAP
    start_repeats: int = DEFAULT_START_REPEATS
    rx_frame_timeout: float = DEFAULT_RX_FRAME_TIMEOUT
    max_rx_frames: int = DEFAULT_MAX_RX_FRAMES
    tx_queue_size: int = DEFAULT_TX_QUEUE
    frame_passes: int = DEFAULT_FRAME_PASSES
    debug: bool = False


@dataclass
class EncodedFrame:
    frame_id: int
    jpeg: bytes
    total_chunks: int
    encoded_at: float = field(default_factory=time.time)


class SDRLink:
    def __init__(self, cfg: StreamConfig, need_tx: bool, need_rx: bool):
        self.cfg = cfg
        self.tx_sdr = None
        self.rx_sdr = None
        self._carrier_cache = {}

        print(f"{ts()} [INIT] Connecting to {cfg.sdr_ip}")
        try:
            if need_rx:
                self.rx_sdr = adi.Pluto(cfg.sdr_ip)
                self.rx_sdr.sample_rate = int(cfg.sample_rate)
                self.rx_sdr.rx_lo = int(cfg.freq)
                self.rx_sdr.rx_buffer_size = int(cfg.rx_buffer_size)
                self.rx_sdr.gain_control_mode_chan0 = "slow_attack"

            if need_tx:
                self.tx_sdr = adi.Pluto(cfg.sdr_ip)
                self.tx_sdr.sample_rate = int(cfg.sample_rate)
                self.tx_sdr.tx_lo = int(cfg.freq)
                self.tx_sdr.tx_hardwaregain_chan0 = int(cfg.tx_gain)
                self.tx_sdr.tx_cyclic_buffer = False
        except Exception as exc:
            print(f"{ts()} [FATAL] SDR connection failed: {exc}")
            sys.exit(1)

        print(f"{ts()} [INIT] SDR ready")

    def packet_samples_for_payload(self, payload_bytes: int) -> int:
        total_bytes = 2 + payload_bytes + 4
        total_bits = total_bytes * 8
        total_symbols = len(BARKER) + total_bits
        return total_symbols * self.cfg.sps

    def _carrier(self, length: int) -> np.ndarray:
        carrier = self._carrier_cache.get(length)
        if carrier is None:
            t = np.arange(length, dtype=np.float32) / self.cfg.sample_rate
            carrier = np.exp(1j * 2 * np.pi * self.cfg.freq_offset * t).astype(np.complex64)
            self._carrier_cache[length] = carrier
        return carrier

    def encode_packet(self, payload: bytes) -> np.ndarray:
        if len(payload) > self.cfg.max_payload_bytes:
            raise ValueError(f"payload too large: {len(payload)} > {self.cfg.max_payload_bytes}")

        frame = struct.pack(">H", len(payload)) + payload
        crc = zlib.crc32(frame) & 0xFFFFFFFF
        framed = frame + struct.pack(">I", crc)
        bits = np.unpackbits(np.frombuffer(framed, dtype=np.uint8))

        state = 1
        dbpsk = np.empty(bits.size, dtype=np.int8)
        for i, bit in enumerate(bits):
            if bit:
                state *= -1
            dbpsk[i] = state

        symbols = np.concatenate((BARKER, dbpsk)).astype(np.float32)
        iq = np.repeat(symbols, self.cfg.sps).astype(np.complex64)
        iq *= self._carrier(iq.size)
        return iq * AMPLITUDE

    def transmit_packet(self, payload: bytes, repeats: int, gap_s: float) -> None:
        if self.tx_sdr is None:
            raise RuntimeError("TX SDR not configured")

        signal = self.encode_packet(payload)
        try:
            self.tx_sdr.tx_destroy_buffer()
        except Exception:
            pass

        for _ in range(repeats):
            self.tx_sdr.tx(signal)
            if gap_s > 0:
                time.sleep(gap_s)

        try:
            self.tx_sdr.tx_destroy_buffer()
        except Exception:
            pass

    def try_receive_packet(self):
        if self.rx_sdr is None:
            raise RuntimeError("RX SDR not configured")

        samples = self.rx_sdr.rx()
        carrier = self._carrier(len(samples))
        shifted = samples * np.conj(carrier)

        corr = np.correlate(shifted, BARKER_UPSAMPLED, mode="valid")
        corr_mag = np.abs(corr)
        peak_idx = int(np.argmax(corr_mag))
        peak_val = float(corr_mag[peak_idx])
        if peak_val < self.cfg.rx_threshold:
            return None, peak_val

        payload_start = peak_idx + len(BARKER_UPSAMPLED)
        max_bytes = 2 + self.cfg.max_payload_bytes + 4
        needed_symbols = 1 + max_bytes * 8
        symbol_stream = shifted[payload_start - self.cfg.sps: payload_start + needed_symbols * self.cfg.sps: self.cfg.sps]
        if len(symbol_stream) < needed_symbols:
            return None, peak_val

        diff = symbol_stream[1:] * np.conj(symbol_stream[:-1])
        bits = (np.real(diff) < 0).astype(np.uint8)
        raw = np.packbits(bits).tobytes()
        if len(raw) < 6:
            return None, peak_val

        payload_len = struct.unpack(">H", raw[:2])[0]
        total_len = 2 + payload_len + 4
        if payload_len == 0 or payload_len > self.cfg.max_payload_bytes or len(raw) < total_len:
            return None, peak_val

        body = raw[: 2 + payload_len]
        crc_rx = struct.unpack(">I", raw[2 + payload_len: total_len])[0]
        crc_expected = zlib.crc32(body) & 0xFFFFFFFF
        if crc_rx != crc_expected:
            return None, peak_val

        return body[2:], peak_val


def chunk_bytes(data: bytes, chunk_size: int):
    for offset in range(0, len(data), chunk_size):
        yield data[offset: offset + chunk_size]


def build_start_packet(width: int, height: int, fps_x100: int, quality: int) -> bytes:
    return struct.pack(">BHHHB", PKT_START, width, height, fps_x100, quality)


def build_end_packet() -> bytes:
    return struct.pack(">B", PKT_END)


def build_chunk_packet(frame_id: int, chunk_idx: int, total_chunks: int, jpeg_size: int, chunk: bytes) -> bytes:
    header = struct.pack(">BIHHI", PKT_CHUNK, frame_id, chunk_idx, total_chunks, jpeg_size)
    return header + chunk


def parse_packet(payload: bytes):
    if not payload:
        return None

    pkt_type = payload[0]
    if pkt_type == PKT_START and len(payload) == struct.calcsize(">BHHHB"):
        width, height, fps_x100, quality = struct.unpack(">HHHB", payload[1:])
        return ("start", {"width": width, "height": height, "fps": fps_x100 / 100.0, "quality": quality})

    if pkt_type == PKT_END and len(payload) == 1:
        return ("end", {})

    if pkt_type == PKT_CHUNK and len(payload) >= struct.calcsize(">BIHHI"):
        frame_id, chunk_idx, total_chunks, jpeg_size = struct.unpack(">IHHI", payload[1:13])
        return (
            "chunk",
            {
                "frame_id": frame_id,
                "chunk_idx": chunk_idx,
                "total_chunks": total_chunks,
                "jpeg_size": jpeg_size,
                "chunk": payload[13:],
            },
        )

    return None


class VideoTx:
    def __init__(self, cfg: StreamConfig, link: SDRLink, video_path: str, loop_video: bool):
        self.cfg = cfg
        self.link = link
        self.video_path = video_path
        self.loop_video = loop_video
        self.running = True
        self.frame_queue = queue.Queue(maxsize=cfg.tx_queue_size)
        self.capture_done = False

        header_len = struct.calcsize(">BIHHI")
        self.chunk_payload_size = cfg.max_payload_bytes - header_len
        if self.chunk_payload_size <= 0:
            raise ValueError("MAX_PAYLOAD_BYTES is too small for the chunk header")

        packet_samples = link.packet_samples_for_payload(cfg.max_payload_bytes)
        chunk_airtime = packet_samples / cfg.sample_rate
        self.chunk_gap = max(cfg.chunk_gap, chunk_airtime * 0.35)

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"{ts()} [FATAL] Could not open video: {self.video_path}")
            self.running = False
            self.capture_done = True
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, int(round(source_fps / self.cfg.target_fps)))
        print(
            f"{ts()} [TX] Source open: {self.video_path} | src_fps={source_fps:.2f} | "
            f"target={self.cfg.target_fps:.2f} | step={frame_step}"
        )

        frame_id = 0
        source_index = 0
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    if self.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        source_index = 0
                        continue
                    break

                if source_index % frame_step != 0:
                    source_index += 1
                    continue

                resized = cv2.resize(frame, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    resized,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)],
                )
                source_index += 1
                if not ok:
                    continue

                jpeg = encoded.tobytes()
                total_chunks = (len(jpeg) + self.chunk_payload_size - 1) // self.chunk_payload_size
                item = EncodedFrame(frame_id=frame_id, jpeg=jpeg, total_chunks=total_chunks)

                while self.running:
                    try:
                        self.frame_queue.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                frame_id += 1
        finally:
            cap.release()
            self.capture_done = True

    def run(self):
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()

        start_packet = build_start_packet(
            self.cfg.width,
            self.cfg.height,
            int(self.cfg.target_fps * 100),
            self.cfg.jpeg_quality,
        )
        self.link.transmit_packet(start_packet, repeats=self.cfg.start_repeats, gap_s=0.01)
        print(
            f"{ts()} [TX] Streaming start | size={self.cfg.width}x{self.cfg.height} "
            f"| q={self.cfg.jpeg_quality} | repeats={self.cfg.chunk_repeats} | gap={self.chunk_gap:.4f}s"
        )

        sent_frames = 0
        sent_chunks = 0
        started = time.time()

        try:
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=0.2)
                except queue.Empty:
                    if self.capture_done:
                        break
                    continue

                chunks = list(chunk_bytes(frame.jpeg, self.chunk_payload_size))
                for frame_pass in range(self.cfg.frame_passes):
                    pass_offset = frame_pass % max(len(chunks), 1)
                    ordered_indices = list(range(pass_offset, len(chunks))) + list(range(0, pass_offset))
                    for chunk_idx in ordered_indices:
                        chunk = chunks[chunk_idx]
                        packet = build_chunk_packet(frame.frame_id, chunk_idx, frame.total_chunks, len(frame.jpeg), chunk)
                        self.link.transmit_packet(packet, repeats=self.cfg.chunk_repeats, gap_s=self.chunk_gap)
                        sent_chunks += 1

                    if self.cfg.debug:
                        print(
                            f"\n{ts()} [TX] frame={frame.frame_id} pass={frame_pass + 1}/{self.cfg.frame_passes} "
                            f"chunks={frame.total_chunks}"
                        )

                sent_frames += 1
                age_ms = (time.time() - frame.encoded_at) * 1000.0
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"\r{ts()} [TX] frame={frame.frame_id} bytes={len(frame.jpeg)} "
                    f"chunks={frame.total_chunks} passes={self.cfg.frame_passes} queue={self.frame_queue.qsize()} "
                    f"fps={sent_frames / elapsed:.2f} age_ms={age_ms:.0f}",
                    end="",
                    flush=True,
                )
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            capture_thread.join(timeout=1.0)
            self.link.transmit_packet(build_end_packet(), repeats=self.cfg.start_repeats, gap_s=0.01)
            print(f"\n{ts()} [TX] Stream ended | frames={sent_frames} chunks={sent_chunks}")


class VideoRx:
    def __init__(self, cfg: StreamConfig, link: SDRLink):
        self.cfg = cfg
        self.link = link
        self.running = True
        self.display_queue = queue.Queue(maxsize=2)
        self.frame_lock = threading.Lock()
        self.frame_state = {}
        self.last_frame_id = -1

    def _drop_stale_frames(self):
        now = time.time()
        stale = []
        with self.frame_lock:
            for frame_id, state in self.frame_state.items():
                if now - state["updated_at"] > self.cfg.rx_frame_timeout or frame_id < self.last_frame_id - 2:
                    stale.append(frame_id)
            for frame_id in stale:
                self.frame_state.pop(frame_id, None)

            while len(self.frame_state) > self.cfg.max_rx_frames:
                oldest = min(self.frame_state, key=lambda fid: self.frame_state[fid]["updated_at"])
                self.frame_state.pop(oldest, None)

    def _queue_frame(self, frame_id: int, jpeg: bytes):
        while not self.display_queue.empty():
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self.display_queue.put_nowait((frame_id, jpeg))
        except queue.Full:
            pass

    def _receiver_loop(self):
        while self.running:
            try:
                payload, peak = self.link.try_receive_packet()
            except Exception as exc:
                print(f"\n{ts()} [RX] SDR read error: {exc}")
                time.sleep(0.2)
                continue

            if payload is None:
                self._drop_stale_frames()
                continue

            parsed = parse_packet(payload)
            if parsed is None:
                if self.cfg.debug:
                    print(f"\n{ts()} [RX] Unknown packet | peak={peak:.0f}")
                continue

            kind, data = parsed
            if kind == "start":
                print(
                    f"\n{ts()} [RX] Stream start | {data['width']}x{data['height']} "
                    f"| fps={data['fps']:.2f} | q={data['quality']}"
                )
                continue

            if kind == "end":
                print(f"\n{ts()} [RX] Stream end")
                continue

            frame_id = data["frame_id"]
            chunk_idx = data["chunk_idx"]
            total_chunks = data["total_chunks"]
            jpeg_size = data["jpeg_size"]
            chunk = data["chunk"]

            if frame_id <= self.last_frame_id:
                continue

            with self.frame_lock:
                state = self.frame_state.get(frame_id)
                if state is None:
                    state = {
                        "total_chunks": total_chunks,
                        "jpeg_size": jpeg_size,
                        "chunks": {},
                        "updated_at": time.time(),
                    }
                    self.frame_state[frame_id] = state

                if total_chunks != state["total_chunks"] or jpeg_size != state["jpeg_size"]:
                    continue

                state["chunks"][chunk_idx] = chunk
                state["updated_at"] = time.time()

                if self.cfg.debug:
                    print(
                        f"\n{ts()} [RX] chunk frame={frame_id} idx={chunk_idx}/{total_chunks - 1} "
                        f"have={len(state['chunks'])}/{total_chunks} bytes={len(chunk)}"
                    )

                if len(state["chunks"]) == total_chunks:
                    assembled = b"".join(state["chunks"].get(i, b"") for i in range(total_chunks))
                    self.frame_state.pop(frame_id, None)
                    if len(assembled) != jpeg_size:
                        continue
                    if frame_id > self.last_frame_id:
                        self.last_frame_id = frame_id
                        if self.cfg.debug:
                            print(f"\n{ts()} [RX] frame complete={frame_id} size={jpeg_size}")
                        self._queue_frame(frame_id, assembled)

            self._drop_stale_frames()

    def run(self):
        rx_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        rx_thread.start()

        cv2.namedWindow("ANTSDR Video RX", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ANTSDR Video RX", 960, 540)
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for SDR video...", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("ANTSDR Video RX", placeholder)

        displayed = 0
        started = time.time()

        try:
            while self.running:
                key = cv2.waitKey(10) & 0xFF
                if key in (27, ord("q")):
                    break

                try:
                    frame_id, jpeg = self.display_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                displayed += 1
                elapsed = max(time.time() - started, 1e-6)
                cv2.imshow("ANTSDR Video RX", frame)
                print(
                    f"\r{ts()} [RX] frame={frame_id} bytes={len(jpeg)} display_fps={displayed / elapsed:.2f}",
                    end="",
                    flush=True,
                )
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            rx_thread.join(timeout=1.0)
            cv2.destroyAllWindows()
            print(f"\n{ts()} [RX] Receiver stopped")


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized one-way ANTSDR video livestreamer for test.mp4")
    parser.add_argument("--mode", choices=["tx", "rx"], required=True, help="Run as video transmitter or receiver")
    parser.add_argument("--sdr-ip", default="ip:192.168.1.11", help="Target SDR URI, for example ip:192.168.1.11")
    parser.add_argument("--file", default=DEFAULT_VIDEO, help="Video file to stream in TX mode")
    parser.add_argument("--loop", action="store_true", help="Loop the input file forever in TX mode")
    parser.add_argument("--freq", type=int, default=FREQ, help="RF center frequency in Hz")
    parser.add_argument("--sample-rate", type=int, default=SAMP_RATE, help="Sample rate in samples/sec")
    parser.add_argument("--sps", type=int, default=SPS, help="Samples per symbol for DBPSK framing")
    parser.add_argument("--freq-offset", type=int, default=FREQ_OFFSET, help="Baseband frequency offset in Hz")
    parser.add_argument("--rx-buffer-size", type=int, default=RX_BUFFER_SIZE, help="RX buffer size")
    parser.add_argument("--rx-threshold", type=int, default=RX_THRESHOLD, help="Correlation threshold")
    parser.add_argument("--max-payload-bytes", type=int, default=MAX_PAYLOAD_BYTES, help="Per-packet payload bytes")
    parser.add_argument("--tx-gain", type=int, default=TX_GAIN, help="TX hardware gain in dB")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="TX resize width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="TX resize height")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality for TX frames")
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS, help="Desired transmitted frame rate")
    parser.add_argument("--chunk-repeats", type=int, default=DEFAULT_CHUNK_REPEATS, help="Repeats per video chunk")
    parser.add_argument("--chunk-gap", type=float, default=DEFAULT_CHUNK_GAP, help="Gap between repeated chunk bursts")
    parser.add_argument("--start-repeats", type=int, default=DEFAULT_START_REPEATS, help="Repeats for start/end packets")
    parser.add_argument("--rx-frame-timeout", type=float, default=DEFAULT_RX_FRAME_TIMEOUT, help="Drop incomplete frames after this many seconds")
    parser.add_argument("--max-rx-frames", type=int, default=DEFAULT_MAX_RX_FRAMES, help="Keep at most this many in-progress RX frames")
    parser.add_argument("--tx-queue-size", type=int, default=DEFAULT_TX_QUEUE, help="Pre-encoded TX frame queue depth")
    parser.add_argument("--frame-passes", type=int, default=DEFAULT_FRAME_PASSES, help="How many full passes of each frame to send before moving on")
    parser.add_argument("--debug", action="store_true", help="Verbose receiver logging")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = StreamConfig(
        sdr_ip=args.sdr_ip,
        freq=args.freq,
        sample_rate=args.sample_rate,
        sps=args.sps,
        freq_offset=args.freq_offset,
        rx_buffer_size=args.rx_buffer_size,
        rx_threshold=args.rx_threshold,
        max_payload_bytes=args.max_payload_bytes,
        tx_gain=args.tx_gain,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
        target_fps=args.target_fps,
        chunk_repeats=args.chunk_repeats,
        chunk_gap=args.chunk_gap,
        start_repeats=args.start_repeats,
        rx_frame_timeout=args.rx_frame_timeout,
        max_rx_frames=args.max_rx_frames,
        tx_queue_size=args.tx_queue_size,
        frame_passes=args.frame_passes,
        debug=args.debug,
    )

    if cfg.max_payload_bytes <= 13:
        print(f"{ts()} [FATAL] --max-payload-bytes must be greater than 13")
        sys.exit(2)

    if args.mode == "tx":
        link = SDRLink(cfg, need_tx=True, need_rx=False)
        VideoTx(cfg, link, args.file, args.loop).run()
    else:
        link = SDRLink(cfg, need_tx=False, need_rx=True)
        VideoRx(cfg, link).run()


if __name__ == "__main__":
    main()
