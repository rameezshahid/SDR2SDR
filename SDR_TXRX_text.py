import threading
import time
import numpy as np
import adi

# =========================
# Configuration
# =========================
SDR_IP = "ip:192.168.1.11"   # change per SDR
FREQ = 5_000_000_000
SAMP_RATE = 1_000_000
SPS = 4
FREQ_OFFSET = 50_000
MAX_MSG_LEN = 200

RX_BUFFER_SIZE = 60000
RX_THRESHOLD = 3000

PREP_MSG = "__PREP__"

BARKER = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.int8)
barker_upsampled = np.repeat(BARKER, SPS)

# =========================
# SDR setup
# =========================
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

# =========================
# Shared state
# =========================
running = True
rx_pause_until = 0.0
prep_window_until = 0.0
last_decoded = ""
last_decoded_time = 0.0
duplicate_window_sec = 1.5

state_lock = threading.Lock()


# =========================
# PHY helpers
# =========================
def encode_message(text: str) -> np.ndarray:
    text = text[:MAX_MSG_LEN]

    length_bits = format(len(text), "08b")
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
    signal = encode_message(text)

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

    needed_symbols = 1 + 8 + (MAX_MSG_LEN * 8)
    symbol_stream = rx_shifted[payload_start - SPS : payload_start + needed_symbols * SPS : SPS]

    if len(symbol_stream) < needed_symbols:
        return None, peak_val

    diff_demod = symbol_stream[1:] * np.conj(symbol_stream[:-1])
    bits = ["1" if np.real(v) < 0 else "0" for v in diff_demod]

    len_bits = "".join(bits[:8])
    msg_len = int(len_bits, 2)

    if not (0 < msg_len <= MAX_MSG_LEN):
        return None, peak_val

    msg_bits = "".join(bits[8:8 + (msg_len * 8)])
    if len(msg_bits) < msg_len * 8:
        return None, peak_val

    chars = [chr(int(msg_bits[i:i+8], 2)) for i in range(0, len(msg_bits), 8)]
    decoded = "".join(chars)

    if not decoded.isprintable():
        return None, peak_val

    return decoded, peak_val


# =========================
# RX thread
# =========================
def receiver_loop():
    global rx_pause_until, prep_window_until, last_decoded, last_decoded_time, running

    while running:
        try:
            samples = rx_sdr.rx()
        except Exception:
            continue

        now = time.time()

        with state_lock:
            paused = now < rx_pause_until

        if paused:
            continue

        decoded, peak_val = try_decode(samples)
        if decoded is None:
            continue

        now = time.time()

        with state_lock:
            is_duplicate = (
                decoded == last_decoded and
                (now - last_decoded_time) < duplicate_window_sec
            )
            if is_duplicate:
                continue

            last_decoded = decoded
            last_decoded_time = now

            if decoded == PREP_MSG:
                prep_window_until = now + 3.0
                print("\n[RX] Incoming message...")
                continue

            if now < prep_window_until:
                print(f"\n[RX] {decoded}")
                prep_window_until = 0.0
            else:
                print(f"\n[RX] {decoded}")

        print("> ", end="", flush=True)


# =========================
# UI / main loop
# =========================
def send_message_flow():
    global rx_pause_until

    msg = input("Message: ").strip()
    if not msg:
        return

    with state_lock:
        rx_pause_until = time.time() + 1.5

    print("[TX] Sending prepare frame...")
    transmit_burst(PREP_MSG, repeats=3, gap_s=0.03)

    time.sleep(0.15)

    print(f"[TX] Sending message: {msg}")
    transmit_burst(msg, repeats=5, gap_s=0.02)

    with state_lock:
        rx_pause_until = time.time() + 1.0


def main():
    global running

    print("--- Half-Duplex SDR Chat ---")
    print("Default mode: listening")
    print("Commands: send, listen, quit")

    rx_thread = threading.Thread(target=receiver_loop, daemon=True)
    rx_thread.start()

    try:
        while True:
            cmd = input("> ").strip().lower()

            if cmd in ("quit", "q", "exit"):
                running = False
                break
            elif cmd in ("listen", "l", ""):
                print("[INFO] Listening...")
            elif cmd in ("send", "s"):
                send_message_flow()
            else:
                print("Use: send, listen, or quit")
    except KeyboardInterrupt:
        running = False
        print("\nExiting...")

    rx_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()