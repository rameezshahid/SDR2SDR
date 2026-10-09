"""Hardware-free loopback test for the DBPSK modem in all three scripts.

Each test modulates a frame with the script's own encoder, drops it into a
simulated RX buffer (random offset, carrier phase rotation, scaling, AWGN),
and checks the script's own decoder recovers the payload and CRC.

No SDR or libiio is needed: `adi` is replaced with a stub before import.

Run:  python tests/loopback_test.py      (or: pytest tests/)
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _FakePluto:
    """Accepts any attribute the scripts set during SDR setup."""

    def __init__(self, uri=None):
        self.uri = uri

    def rx(self):
        raise RuntimeError("fake SDR has no RX samples")

    def tx(self, _samples):
        pass

    def tx_destroy_buffer(self):
        pass


sys.modules["adi"] = types.SimpleNamespace(Pluto=_FakePluto)

RNG = np.random.default_rng(1234)


def channel(tx: np.ndarray, pad: int = 300, noise_std: float = 50.0) -> np.ndarray:
    """Simulate the air: random start offset, unknown carrier phase, gain loss, noise."""
    lead = int(RNG.integers(20, pad))
    phase = np.exp(1j * RNG.uniform(0, 2 * np.pi))
    buf = np.zeros(lead + tx.size + pad, dtype=np.complex64)
    buf[lead:lead + tx.size] = tx * phase * 0.25
    noise = (RNG.normal(0, noise_std, buf.size) + 1j * RNG.normal(0, noise_std, buf.size)).astype(np.complex64)
    return buf + noise


def test_text_chat_roundtrip():
    mod = importlib.import_module("SDR_TXRX_text")
    for msg in ["hello", "Half-duplex SDR chat over DBPSK!", "x" * mod.MAX_MSG_LEN]:
        decoded, peak = mod.try_decode(channel(mod.encode_message(msg)))
        assert decoded == msg, f"expected {msg!r}, got {decoded!r} (peak={peak:.0f})"


def test_file_transfer_frames_with_crc():
    mod = importlib.import_module("FILE_TEXT_SDR_TXRX")
    frames = [
        "plain text message",
        mod.IMG_PREP_MSG,
        f"{mod.IMG_START}|photo.jpg|12",
        f"{mod.IMG_CHUNK}|7|" + "QUJD" * (mod.IMG_CHUNK_B64_LEN // 4),
        f"{mod.IMG_NACK}|1,4,9",
    ]
    for payload in frames:
        framed = mod._frame_add_crc(payload)
        decoded, peak = mod.try_decode(channel(mod.encode_message(framed)))
        assert decoded == payload, f"expected {payload[:40]!r}, got {decoded!r} (peak={peak:.0f})"


def test_file_transfer_rejects_corrupted_crc():
    mod = importlib.import_module("FILE_TEXT_SDR_TXRX")
    framed = mod._frame_add_crc("tamper me")
    corrupted = framed[:-1] + ("0" if framed[-1] != "0" else "1")
    decoded, _ = mod.try_decode(channel(mod.encode_message(corrupted)))
    assert decoded is None, "frame with a bad CRC must be dropped"


def test_video_stream_packets():
    mod = importlib.import_module("STREAM_FILE_TEXT_SDR_TXRX")
    cfg = mod.StreamConfig(sdr_ip="ip:fake")
    link = mod.SDRLink.__new__(mod.SDRLink)  # skip __init__: no hardware
    link.cfg, link.tx_sdr, link._carrier_cache = cfg, None, {}

    jpeg_chunk = bytes(RNG.integers(0, 256, cfg.max_payload_bytes - 13, dtype=np.uint8))
    packets = [
        mod.build_start_packet(160, 90, 150, 18),
        mod.build_chunk_packet(frame_id=42, chunk_idx=3, total_chunks=9, jpeg_size=700, chunk=jpeg_chunk),
        mod.build_end_packet(),
    ]
    # The decoder always reads a max-size packet window after the sync word,
    # so the simulated buffer needs at least that many samples after it.
    tail = link.packet_samples_for_payload(cfg.max_payload_bytes)
    for payload in packets:
        link.rx_sdr = types.SimpleNamespace(rx=lambda p=payload: channel(link.encode_packet(p), pad=tail))
        received, peak = link.try_receive_packet()
        assert received == payload, f"packet mismatch (peak={peak:.0f})"

    kind, info = mod.parse_packet(packets[1])
    assert kind == "chunk" and info["frame_id"] == 42 and info["chunk"] == jpeg_chunk


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} loopback tests passed.")
