# SDR2SDR

A point-to-point radio link between two software-defined radios (ADALM-Pluto or Pluto-compatible boards such as ANTSDR), written from scratch in Python. It covers the physical layer (DBPSK modulation, preamble sync, framing) and the link layer (CRC, retransmission), with three applications built on top:

| Script | What it does |
|---|---|
| `SDR_TXRX_text.py` | Half-duplex **text chat** between two radios |
| `FILE_TEXT_SDR_TXRX.py` | Chat plus **image/file transfer** with CRC-32 checks and NACK-based selective retransmission |
| `STREAM_FILE_TEXT_SDR_TXRX.py` | One-way **live video streaming**: frames are JPEG-compressed, chunked, sent over the air, and reassembled in a viewer window |

## How it works

```text
TX:  bytes -> frame (length + payload + CRC) -> differential encoding -> Barker-13 preamble
         -> upsample (4 samples/symbol) -> shift to +50 kHz IF -> Pluto TX @ 5 GHz

RX:  Pluto RX -> shift IF back to baseband -> correlate with Barker-13 (frame sync)
         -> sample symbols -> differential demod (phase-invariant) -> length + CRC check -> payload
```

- **Modulation:** DBPSK at 1 MS/s with 4 samples/symbol (250 ksymbol/s). Differential encoding means the receiver doesn't need carrier phase recovery.
- **Synchronization:** a 13-chip Barker code marks the start of each frame. The receiver finds it by cross-correlation and ignores anything below a threshold.
- **Integrity:** the chat sends each message several times and drops duplicates. File transfer and video append a CRC-32 and discard corrupted frames.
- **Half-duplex control:** while transmitting, the receiver pauses so a radio doesn't decode its own signal. A short `__PREP__` frame warns the other side that a message is coming.
- **Reliable file transfer:** an image is base64-encoded into numbered chunks (`START` / `CHUNK` / `END`). The receiver replies with a NACK listing missing chunks, and the sender retransmits only those, for up to 5 rounds. Received files are saved to `received_images/`.
- **Video streaming:** frames are resized (default 160×90), JPEG-encoded at low quality, and split into 96-byte binary packets. Each frame is sent in several passes with a rotating chunk order, so a lost packet is likely caught on the next pass. The receiver reassembles frames, drops stale or incomplete ones, and shows the newest complete frame.

## Hardware

- 2 × ADALM-Pluto (or a Pluto-compatible SDR), each reachable over the network. The defaults assume `192.168.1.10` and `192.168.1.11`.
- Antennas, or for bench testing **SMA cables with 30–40 dB attenuators** between the radios.

> ⚠️ **RF regulations:** the default carrier is **5.000 GHz**. Transmitting over the air on any frequency may require a licence where you are. For bench tests use cabled connections with attenuators, or change `FREQ` to a band you're allowed to use.

## Setup

1. Install the **libiio** C library (needed by `pyadi-iio` to talk to the Pluto). Download it from [analogdevicesinc/libiio releases](https://github.com/analogdevicesinc/libiio/releases). On Debian/Ubuntu you can run `sudo apt install libiio0` instead.
2. Install the Python dependencies (Python 3.10+):

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

3. Check each radio is reachable with `iio_info -u ip:192.168.1.10`.

## Testing without hardware

`tests/loopback_test.py` runs every script's modulator and demodulator against a simulated channel: random timing offset, unknown carrier phase, attenuation and noise. It checks text, file-transfer frames (including rejection of a corrupted CRC) and video packets. It needs no SDR and no libiio:

```bash
python tests/loopback_test.py
```

```text
PASS  test_file_transfer_frames_with_crc
PASS  test_file_transfer_rejects_corrupted_crc
PASS  test_text_chat_roundtrip
PASS  test_video_stream_packets

All 4 loopback tests passed.
```

## Running with two radios

### Text chat / file transfer

In `SDR_TXRX_text.py` or `FILE_TEXT_SDR_TXRX.py`, set `SDR_IP` at the top to the local radio (`ip:192.168.1.10` on one machine, `ip:192.168.1.11` on the other). Then run the same script on both sides:

```bash
python FILE_TEXT_SDR_TXRX.py
```

| Command | Action |
|---|---|
| `send` / `s` | Type a message and transmit it |
| `sendimg` / `img` | Send a file (image path); the other side saves it to `received_images/` |
| `listen` / `l` | Keep listening (the default) |
| `quit` / `q` | Exit |

Set `DEBUG = True` to print raw decoded frames before the CRC check.

### Video streaming

The receiver opens a viewer window. The transmitter streams a local video file (not included; use any `.mp4`).

```bash
# Receiver
python STREAM_FILE_TEXT_SDR_TXRX.py --mode rx --sdr-ip ip:192.168.1.10

# Transmitter
python STREAM_FILE_TEXT_SDR_TXRX.py --mode tx --sdr-ip ip:192.168.1.11 --file my_video.mp4 --loop
```

Useful options: `--width/--height`, `--jpeg-quality`, `--target-fps`, `--frame-passes`, `--chunk-repeats`, `--freq`, `--tx-gain`, and `--debug`. Run `--help` for the full list. Press `q` or `Esc` in the viewer to stop.

## Limitations

- Text frames carry 8-bit characters, so use plain ASCII. Non-Latin text or emoji won't decode correctly.
- There is no encryption or authentication. Anyone tuned to the same frequency can receive or inject frames.
- The links are half-duplex (chat/file) or one-way (video), on a fixed carrier with a fixed correlation threshold. You may need to adjust `RX_THRESHOLD` and TX gain for your setup.
