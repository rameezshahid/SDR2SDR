import threading
import time
import numpy as np
import adi
import os
import base64
import sys
import zlib

# =========================
# Configuration
# =========================
DEBUG = False   # Set True to print raw decoded strings before CRC check
SDR_IP = "ip:192.168.1.11"   # Change per machine (.11 or .10)
FREQ = 5_000_000_000
SAMP_RATE = 1_000_000
SPS = 4
FREQ_OFFSET = 50_000
MAX_MSG_LEN = 400

RX_BUFFER_SIZE = 131072
RX_THRESHOLD = 2000

PREP_MSG = "__PREP__"

# Image transfer markers
IMG_PREP_MSG  = "__IP__"
IMG_START     = "__IS__"
IMG_CHUNK     = "__IC__"
IMG_END       = "__IE__"
IMG_ABORT     = "__IA__"
IMG_NACK      = "__IN__"   # retransmit request:  __IN__|idx1,idx2,...
RECV_IMG_DIR  = "received_images"
IMG_CHUNK_B64_LEN = 370  # 11 prefix + 370 payload + 9 CRC = 390, under MAX_MSG_LEN=400

# CRC separator appended to every transmitted frame: "payload~XXXXXXXX"
CRC_SEP = "~"

BARKER = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.int8)
barker_upsampled = np.repeat(BARKER, SPS)

def ts():
    """Return current time as HH:MM:SS.mmm for log prefixing."""
    t = time.time()
    ms = int((t % 1) * 1000)
    return time.strftime(f"%H:%M:%S.") + f"{ms:03d}"

# =========================
# SDR setup
# =========================
print(f"{ts()} [INIT] Connecting to SDR at {SDR_IP} ...")
try:
    rx_sdr = adi.Pluto(SDR_IP)
    rx_sdr.sample_rate = int(SAMP_RATE)
    rx_sdr.rx_lo = int(FREQ)
    rx_sdr.rx_buffer_size = RX_BUFFER_SIZE
    rx_sdr.gain_control_mode_chan0 = "slow_attack"

    tx_sdr = adi.Pluto(SDR_IP)
    tx_sdr.sample_rate = int(SAMP_RATE)
    tx_sdr.tx_lo = int(FREQ)
    tx_sdr.tx_hardwaregain_chan0 = -10
    tx_sdr.tx_cyclic_buffer = False

    print(f"{ts()} [INIT] SDR connected OK.")
except Exception as e:
    print(f"\n{ts()} [FATAL] Could not connect to SDR at {SDR_IP}")
    print(f"        Error: {e}")
    print(f"        Check the IP and that the Pluto is reachable.")
    sys.exit(1)

# =========================
# Shared state
# =========================
running = True
rx_pause_until = 0.0
prep_window_until = 0.0
last_decoded = ""
last_decoded_time = 0.0
duplicate_window_sec = 1.5

# image RX state
image_rx_active = False
image_rx_name = ""
image_rx_total_chunks = 0
image_rx_chunks = {}
image_rx_started_at = 0.0

# NACK state — set by RX handler, consumed by a retransmit thread
nack_pending = []          # list of chunk indices to retransmit
nack_lock = threading.Lock()

state_lock = threading.Lock()


# =========================
# CRC helpers
# =========================
def _frame_add_crc(payload: str) -> str:
    """Append ~CRC32HEX to a payload string."""
    crc = zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF
    return f"{payload}{CRC_SEP}{crc:08X}"


def _frame_check_crc(raw: str):
    """
    Strip and verify the trailing ~CRC32HEX from a received string.
    Returns (payload, True) on success, (None, False) on failure.
    """
    sep_idx = raw.rfind(CRC_SEP)
    if sep_idx == -1:
        return None, False
    payload = raw[:sep_idx]
    crc_str = raw[sep_idx + len(CRC_SEP):]
    if len(crc_str) != 8:
        return None, False
    try:
        received_crc = int(crc_str, 16)
    except ValueError:
        return None, False
    expected_crc = zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF
    if received_crc != expected_crc:
        return None, False
    return payload, True


# =========================
# PHY helpers
# =========================
def encode_message(text: str) -> np.ndarray:
    text = text[:MAX_MSG_LEN]

    length_bits = format(len(text), "016b")  # 16-bit length field, supports up to 65535 chars
    data_bits = "".join(format(ord(ch), "08b") for ch in text)

    padding_len = (MAX_MSG_LEN * 8) - (len(text) * 8)
    all_bits = [int(b) for b in (length_bits + data_bits + ("0" * padding_len))]

    dbits = []
    curr = 1
    for b in all_bits:
        if b == 1:
            curr *= -1
        dbits.append(curr)

    full_frame = np.concatenate((BARKER, np.array(dbits, dtype=np.int8)))
    iq = np.repeat(full_frame, SPS).astype(np.complex64)

    t = np.arange(len(iq)) / SAMP_RATE
    iq *= np.exp(1j * 2 * np.pi * FREQ_OFFSET * t)

    return (iq * (2**14)).astype(np.complex64)


def transmit_burst(text: str, repeats: int = 5, gap_s: float = 0.02) -> None:
    """Truncate payload to MAX_MSG_LEN-9, append CRC, then transmit.
    Truncation must happen BEFORE CRC is appended so the 9-char suffix
    (~XXXXXXXX) is never cut off by encode_message's length cap.
    """
    text = text[: MAX_MSG_LEN - 9]   # reserve 9 chars for ~XXXXXXXX
    framed = _frame_add_crc(text)     # now len(framed) <= MAX_MSG_LEN (200)
    signal = encode_message(framed)

    try:
        tx_sdr.tx_destroy_buffer()
    except Exception:
        pass

    for _ in range(repeats):
        tx_sdr.tx(signal)
        time.sleep(gap_s)

    try:
        tx_sdr.tx_destroy_buffer()
    except Exception:
        pass


def try_decode(rx_samples: np.ndarray):
    t = np.arange(len(rx_samples)) / SAMP_RATE
    rx_shifted = rx_samples * np.exp(-1j * 2 * np.pi * FREQ_OFFSET * t)

    corr = np.correlate(rx_shifted, barker_upsampled, mode="valid")
    mag_corr = np.abs(corr)
    peak_idx = int(np.argmax(mag_corr))
    peak_val = float(mag_corr[peak_idx])

    if peak_val <= RX_THRESHOLD:
        return None, peak_val

    payload_start = peak_idx + len(barker_upsampled)

    needed_symbols = 1 + 16 + (MAX_MSG_LEN * 8)  # 16-bit length field
    symbol_stream = rx_shifted[payload_start - SPS : payload_start + needed_symbols * SPS : SPS]

    if len(symbol_stream) < needed_symbols:
        return None, peak_val

    diff_demod = symbol_stream[1:] * np.conj(symbol_stream[:-1])
    bits = ["1" if np.real(v) < 0 else "0" for v in diff_demod]

    len_bits = "".join(bits[:16])  # 16-bit length field
    msg_len = int(len_bits, 2)

    if not (0 < msg_len <= MAX_MSG_LEN):
        return None, peak_val

    msg_bits = "".join(bits[16:16 + (msg_len * 8)])
    if len(msg_bits) < msg_len * 8:
        return None, peak_val

    chars = [chr(int(msg_bits[i:i+8], 2)) for i in range(0, len(msg_bits), 8)]
    raw = "".join(chars)

    if not raw.isprintable():
        return None, peak_val

    # Strip and verify CRC — reject frame if CRC fails
    if DEBUG:
        print(f"\n{ts()} [DBG] raw='{raw[:60]}' peak={peak_val:.0f}", flush=True)
    payload, ok = _frame_check_crc(raw)
    if not ok:
        if DEBUG:
            print(f"{ts()} [DBG] CRC FAIL on: '{raw[:60]}'", flush=True)
        return None, peak_val

    return payload, peak_val


# =========================
# Image helpers
# =========================
def _safe_image_name(name: str) -> str:
    base = os.path.basename(name).strip()
    if not base:
        base = f"image_{int(time.time())}.bin"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)


def _reset_image_rx_state():
    global image_rx_active, image_rx_name, image_rx_total_chunks
    global image_rx_chunks, image_rx_started_at, last_decoded, last_decoded_time

    image_rx_active = False
    image_rx_name = ""
    image_rx_total_chunks = 0
    image_rx_chunks = {}
    image_rx_started_at = 0.0
    last_decoded = ""
    last_decoded_time = 0.0


def _handle_image_frame(decoded: str) -> bool:
    # Called OUTSIDE state_lock — acquires it internally as needed.
    global image_rx_active, image_rx_name, image_rx_total_chunks
    global image_rx_chunks, image_rx_started_at

    if decoded == IMG_PREP_MSG:
        print(f"\n{ts()} [RX] Incoming image signal received...")
        _reset_image_rx_state()
        return True

    if decoded.startswith(IMG_START + "|"):
        _reset_image_rx_state()

        parts = decoded.split("|", 2)
        if len(parts) != 3:
            print("\n[DEBUG] Malformed header")
            return True

        try:
            name = _safe_image_name(parts[1])
            total_chunks = int(parts[2])

            if not (0 < total_chunks < 10000):
                print(f"\n{ts()} [DEBUG] Invalid chunk count: {total_chunks}")
                return True

            with state_lock:
                image_rx_active = True
                image_rx_name = name
                image_rx_total_chunks = total_chunks
                image_rx_chunks = {}
                image_rx_started_at = time.time()

            os.makedirs(RECV_IMG_DIR, exist_ok=True)
            print(f"\n{ts()} [RX] Session started: {name}")
            print(f"{ts()} [RX] Expected chunks: {total_chunks}")

        except (ValueError, IndexError) as e:
            print(f"\n{ts()} [DEBUG] Failed to parse header: {e}")
            _reset_image_rx_state()

        return True

    if decoded.startswith(IMG_CHUNK + "|"):
        if not image_rx_active:
            return True   # silently ignore — no session active

        parts = decoded.split("|", 2)
        if len(parts) != 3:
            return True

        try:
            idx = int(parts[1])   # will raise ValueError if corrupted
            payload = parts[2]

            if not (0 <= idx < image_rx_total_chunks):
                return True

            if idx not in image_rx_chunks:
                with state_lock:
                    image_rx_chunks[idx] = payload

                received_count = len(image_rx_chunks)
                percent = (received_count / image_rx_total_chunks) * 100
                print(f"\r{ts()} [RX] Progress: {received_count}/{image_rx_total_chunks} [{percent:.1f}%]", end="", flush=True)

        except (ValueError, IndexError):
            pass  # corrupted index — CRC should have caught this, but just in case

        return True

    if decoded.startswith(IMG_END + "|"):
        if not image_rx_active:
            return True

        parts = decoded.split("|", 1)
        name = _safe_image_name(parts[1]) if len(parts) == 2 else image_rx_name

        missing = [i for i in range(image_rx_total_chunks) if i not in image_rx_chunks]

        if missing:
            # Send a NACK listing the missing chunk indices so TX can retransmit
            print(f"\n{ts()} [RX] END received — missing {len(missing)} chunks, sending NACK...")
            with nack_lock:
                nack_pending.clear()
                nack_pending.extend(missing)
            # Transmit NACK immediately (comma-separated indices, batched to fit MAX_MSG_LEN)
            _send_nack(missing)
            return True

        # All chunks received — assemble and save
        try:
            payload_b64 = "".join(image_rx_chunks[i] for i in range(image_rx_total_chunks))
            image_bytes = base64.b64decode(payload_b64.encode("ascii"))
            save_path = os.path.join(RECV_IMG_DIR, name)
            with open(save_path, "wb") as f:
                f.write(image_bytes)
            elapsed = time.time() - image_rx_started_at if image_rx_started_at else 0.0
            print(f"\n{ts()} [RX] Image saved: {save_path} ({len(image_bytes)} bytes, {elapsed:.1f}s)")
        except Exception as e:
            print(f"\n{ts()} [RX] Failed to save image: {e}")
        finally:
            _reset_image_rx_state()
        return True

    if decoded.startswith(IMG_NACK + "|"):
        # The other side is asking us to retransmit missing chunks.
        # This is handled in receiver_loop — return False so it falls through.
        return False

    if decoded == IMG_ABORT:
        print(f"\n{ts()} [RX] Image transfer aborted")
        _reset_image_rx_state()
        return True

    return False


def _send_nack(missing_indices: list) -> None:
    """Transmit NACK frames for missing chunk indices (batched to fit MAX_MSG_LEN).
    Pauses local RX briefly so we don't decode our own NACK transmission.
    """
    global rx_pause_until
    # Each index up to 4 digits + comma = 5 chars, 15 per frame:
    # 15*5=75 + "__IN__|" (7) + "~XXXXXXXX" (9) = 91 chars, under 200
    BATCH = 15
    total_time = (len(missing_indices) / BATCH) * 5 * 0.03 + 1.0
    with state_lock:
        rx_pause_until = time.time() + total_time
    for i in range(0, len(missing_indices), BATCH):
        batch = missing_indices[i:i + BATCH]
        nack_frame = f"{IMG_NACK}|{','.join(str(x) for x in batch)}"
        transmit_burst(nack_frame, repeats=5, gap_s=0.03)
    print(f"{ts()} [RX] NACK sent for {len(missing_indices)} missing chunks")


# =========================
# TX flows
# =========================
def send_message_flow():
    global rx_pause_until

    msg = input("Message: ").strip()
    if not msg:
        return

    with state_lock:
        rx_pause_until = time.time() + 1.5

    print(f"{ts()} [TX] Sending prepare frame...")
    transmit_burst(PREP_MSG, repeats=3, gap_s=0.03)
    time.sleep(0.15)

    print(f"{ts()} [TX] Sending message: {msg}")
    transmit_burst(msg, repeats=5, gap_s=0.02)

    with state_lock:
        rx_pause_until = time.time() + 1.0


def send_image_flow():
    global rx_pause_until

    path = input("Image path: ").strip().strip('"')
    if not path:
        return
    if not os.path.isfile(path):
        print(f"{ts()} [ERR] File not found")
        return

    try:
        with open(path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"{ts()} [ERR] Could not read file: {e}")
        return

    b64 = base64.b64encode(image_bytes).decode("ascii")
    chunks = [b64[i:i + IMG_CHUNK_B64_LEN] for i in range(0, len(b64), IMG_CHUNK_B64_LEN)]
    name = _safe_image_name(os.path.basename(path))
    total_chunks = len(chunks)

    if total_chunks == 0:
        print(f"{ts()} [ERR] Empty file")
        return

    # Store chunks so retransmit thread can access them
    chunk_map = {idx: chunk for idx, chunk in enumerate(chunks)}

    with state_lock:
        rx_pause_until = time.time() + 2.5

    print(f"{ts()} [TX] Sending image prepare frame...")
    transmit_burst(IMG_PREP_MSG, repeats=3, gap_s=0.03)
    time.sleep(0.15)

    start_frame = f"{IMG_START}|{name}|{total_chunks}"
    print(f"{ts()} [TX] Sending image header: {name} ({len(image_bytes)} bytes, {total_chunks} chunks)")
    transmit_burst(start_frame, repeats=8, gap_s=0.03)
    time.sleep(0.2)

    _transmit_chunks(chunk_map, total_chunks)

    end_frame = f"{IMG_END}|{name}"

    # Unlock RX BEFORE sending END so we can hear the NACK that comes back.
    with state_lock:
        rx_pause_until = 0.0

    # Fewer repeats = shorter deaf window while transmitting END.
    # The RX fires its NACK immediately after receiving END, so we must
    # stop transmitting quickly and start listening.
    transmit_burst(end_frame, repeats=5, gap_s=0.02)

    print(f"\n{ts()} [TX] Initial transmission complete. Waiting for NACK...")
    # Give RX time to process END, transmit NACK (~0.5s), and our
    # receiver thread time to decode it before the poll loop checks.
    time.sleep(3.0)

    # Wait for NACK and retransmit up to 5 rounds
    for retry in range(5):
        # Listen for NACK for up to 10 seconds
        nack_deadline = time.time() + 20.0
        got_nack = False
        while time.time() < nack_deadline:
            with nack_lock:
                if nack_pending:
                    missing = list(nack_pending)
                    nack_pending.clear()
                    got_nack = True
                    break
            time.sleep(0.1)

        if not got_nack:
            print(f"{ts()} [TX] No NACK received - transfer complete.")
            break

        print(f"{ts()} [TX] Retransmit round {retry + 1}: resending {len(missing)} chunks...")

        # Clear nack_pending NOW — after consuming missing, before retransmitting.
        # Any new NACK that arrives during retransmit will be freshly appended.
        with nack_lock:
            nack_pending.clear()

        retransmit_map = {idx: chunk_map[idx] for idx in missing if idx in chunk_map}
        _transmit_chunks(retransmit_map, total_chunks, is_retry=True)

        # Unlock RX BEFORE sending END so we can hear the NACK that comes right after
        with state_lock:
            rx_pause_until = 0.0

        transmit_burst(end_frame, repeats=5, gap_s=0.02)
        print(f"{ts()} [TX] Retransmit round {retry + 1} END sent.")
        time.sleep(3.0)  # give RX time to process END and transmit NACK back

    print(f"{ts()} [TX] Image sent: {name}")
    with state_lock:
        rx_pause_until = 0.0


def _transmit_chunks(chunk_map: dict, total_chunks: int, is_retry: bool = False) -> None:
    """Transmit a dict of {idx: payload} chunks."""
    indices = sorted(chunk_map.keys())
    for i, idx in enumerate(indices):
        frame = f"{IMG_CHUNK}|{idx}|{chunk_map[idx]}"
        transmit_burst(frame, repeats=4, gap_s=0.02)
        time.sleep(0.01)
        if i % 10 == 0 or i == len(indices) - 1:
            label = "Retry" if is_retry else "Chunk"
            print(f"{ts()} [TX] {label} {idx + 1}/{total_chunks}")


# =========================
# RX thread
# =========================
def receiver_loop():
    global rx_pause_until, prep_window_until, last_decoded, last_decoded_time, running

    while running:
        try:
            samples = rx_sdr.rx()
        except Exception as e:
            print(f"\n{ts()} [RX ERROR] {e}")
            time.sleep(0.5)
            continue

        now = time.time()

        with state_lock:
            paused = now < rx_pause_until

        if paused:
            continue

        # try_decode now returns None if CRC fails — corrupted frames are dropped here
        decoded, peak_val = try_decode(samples)
        if decoded is None:
            continue

        now = time.time()

        # Duplicate detection
        with state_lock:
            is_img_data = any(x in decoded for x in [IMG_CHUNK, IMG_START, IMG_END, IMG_NACK])
            effective_window = 0.05 if is_img_data else duplicate_window_sec
            is_duplicate = (
                decoded == last_decoded and
                (now - last_decoded_time) < effective_window
            )
            if not is_duplicate:
                last_decoded = decoded
                last_decoded_time = now

        if is_duplicate:
            continue

        # NACK received — the other side wants us to retransmit missing chunks.
        # We store them in nack_pending; send_image_flow() polls that list.
        if decoded.startswith(IMG_NACK + "|"):
            parts = decoded.split("|", 1)
            if len(parts) == 2:
                try:
                    indices = [int(x) for x in parts[1].split(",") if x.strip()]
                    with nack_lock:
                        for idx in indices:
                            if idx not in nack_pending:
                                nack_pending.append(idx)
                    print(f"\n{ts()} [RX] NACK received: {len(indices)} chunks requested")
                    print("> ", end="", flush=True)
                except ValueError:
                    pass
            continue

        if decoded == PREP_MSG:
            with state_lock:
                prep_window_until = now + 3.0
            print(f"\n{ts()} [RX] Incoming message...")
            print("> ", end="", flush=True)
            continue

        if _handle_image_frame(decoded):
            print("> ", end="", flush=True)
            continue

        with state_lock:
            in_prep_window = now < prep_window_until
            if in_prep_window:
                prep_window_until = 0.0

        print(f"\n{ts()} [RX] {decoded}")
        print("> ", end="", flush=True)


# =========================
# Main
# =========================
def main():
    global running

    print("--- Half-Duplex SDR Chat ---")
    print("Default mode: listening")
    print("Commands: send, sendimg, listen, quit")

    rx_thread = threading.Thread(target=receiver_loop, daemon=True)
    rx_thread.start()

    try:
        while True:
            cmd = input("> ").strip().lower()

            if cmd in ("quit", "q", "exit"):
                running = False
                break
            elif cmd in ("listen", "l", ""):
                print(f"{ts()} [INFO] Listening...")
            elif cmd in ("send", "s"):
                send_message_flow()
            elif cmd in ("sendimg", "img", "image"):
                send_image_flow()
            else:
                print("Commands: send, sendimg, listen, quit")
    except KeyboardInterrupt:
        running = False
        print("\nExiting...")

    rx_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()