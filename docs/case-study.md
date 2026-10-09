# SDR2SDR: product case study

*How a radio link was scoped, built in increments, and shaped by measurable constraints.*

## TL;DR

- **Goal:** a working wireless data link between two low-cost software-defined radios, built from first principles. Every layer is hand-written, from modulation to retransmission, with no off-the-shelf protocol stack.
- **Delivery:** three increments, each building on the last: text chat → reliable file transfer → live video.
- **Constraint-driven design:** every major feature decision traces back to an airtime budget. The streaming protocol moves about **1.9× more data per transmission** than the file-transfer protocol it replaced.
- **Quality:** an automated loopback test suite validates all three protocols against simulated radio impairments, without hardware.

## The problem

Two radios, one shared channel, and nothing in between: no Wi-Fi, no TCP/IP, no error correction. The question was how far a simple, transparent design can go, and what each step up in capability costs.

That framing made the work a series of product decisions rather than a single build. Each new payload type (text, files, video) had different requirements, and each required a deliberate trade-off between **reliability**, **speed** and **complexity**.

## Approach: ship in increments

| Increment | Scope | Requirement it had to meet | New capability added |
|---|---|---|---|
| **v1 · Text chat** | Half-duplex messaging (`SDR_TXRX_text.py`) | Prove the link works end to end | DBPSK modem, Barker-13 sync, repeat-and-deduplicate delivery |
| **v2 · File transfer** | Chat plus images (`FILE_TEXT_SDR_TXRX.py`) | Files must arrive **intact**; one bad byte corrupts the image | CRC-32 on every frame, chunking, NACK-based selective retransmission |
| **v3 · Video stream** | One-way live video (`STREAM_FILE_TEXT_SDR_TXRX.py`) | Frames must arrive **in time**; a late frame is worthless | Binary packet format, multi-pass sending, stale-frame dropping |

Each increment was a usable product on its own, and each reused what the previous one had proven. Moving from v2 to v3 meant re-examining v2's assumptions, not just adding features.

## Key trade-off decisions

| Decision | Options considered | Chosen | Why | Cost accepted |
|---|---|---|---|---|
| Chat reliability | Acknowledgements vs blind repetition | Send each message 5× and drop duplicates | No return channel needed, simplest to get right | Uses 5× the airtime |
| File integrity | Resend everything vs resend only what's missing | CRC-32 per frame plus a NACK listing missing chunks; up to 5 retry rounds | Large files make blanket resends too slow | More protocol states to manage |
| Half-duplex timing | Fixed turn-taking vs event-driven | Mute RX while transmitting, but reopen it *before* sending the end marker so the NACK reply isn't missed | A radio can't hear while it talks; timing is everything | Short blind windows where frames can be lost |
| Video: completeness vs latency | Retransmit lost packets vs move on | **Latency wins**: drop stale or incomplete frames and show the newest complete one | Live video is useless late; one-way links can't ask for resends anyway | Occasional skipped frames |
| Video: recovering losses without a return channel | — | Send each frame in 3 passes with a rotating chunk order | A packet lost in one pass is likely caught in the next | 3× airtime per frame |
| Video: wire format | Reuse v2's text/base64 frames vs a new binary format | **New binary packets** sized to the payload | v2 frames are always padded to full length, and base64 adds 33% overhead | Two protocols to maintain |

## Capacity planning (the numbers behind the decisions)

Theoretical airtime budgets, calculated from each script's parameters (1 MS/s, 4 samples/symbol). They exclude host and USB latency, so real-world figures will be lower. Measuring those on hardware is the first item on the roadmap.

| Protocol | Airtime per frame | Send cycle | Effective throughput |
|---|---|---|---|
| v1 chat | 6.5 ms | ~390 ms per message, whatever its length | n/a (interactive) |
| v2 file transfer | 12.9 ms | 142 ms per 277-byte chunk (4 repeats) | **~2.0 kB/s** (a 100 kB image takes about 51 s) |
| v3 video | 3.3 ms | 22 ms per 83-byte chunk (3 repeats) | **~3.8 kB/s per pass**, ~1.3 kB/s of unique data at 3 passes |

What the numbers drove:

- **The v3 redesign was justified by data.** Per transmitted copy, the binary format carries about **1.9× the throughput** of v2's text frames: 0.80 vs 0.69 useful bits per symbol, plus shorter frames and fewer repeats.
- **The video settings follow from the budget.** At 3 passes, the 1.5 fps target is only reachable if each frame compresses to about 800 bytes, so the defaults are 160×90 resolution at JPEG quality 18. Frames of 1.2 kB drop it to about 1 fps.
- **v1's fixed-length frames were fine for chat but not beyond.** A 5-character message costs the same airtime as a 200-character one, which is acceptable for chat and a key reason v2 and v3 changed the framing.

## Quality and validation

Testing radio software usually needs two radios, a lab and a clear channel. To remove that bottleneck, I added `tests/loopback_test.py`. It runs each script's own modulator and demodulator through a simulated channel with random timing offset, unknown carrier phase, signal loss and noise. The suite also confirms that a frame with a bad CRC is rejected.

- **4 automated tests**, all passing, covering all three protocols.
- **No hardware or drivers needed**, so it can run on any laptop or in CI.
- **It has already found an issue.** Short video control packets (start/end) are only decoded if a full packet's worth of samples follows them in the receive buffer, so one arriving late in a buffer can be missed. The protocol currently masks this with repeats; it's logged in the backlog below.

## Risks and compliance

- **Spectrum regulation.** The default carrier is 5.000 GHz. In Canada, radio spectrum is regulated by **Innovation, Science and Economic Development Canada (ISED)**, and over-the-air transmission needs the right authorization (for example, an amateur radio certificate on amateur bands). The project is designed for **cabled bench testing with attenuators**, and the frequency is a single parameter that can be changed.
- **Security.** Frames are unencrypted and unauthenticated. That's acceptable for a lab prototype, but a blocker for any real deployment.
- **Environment sensitivity.** Detection thresholds and transmit gain are fixed values and may need tuning for each setup.

## What's next: prioritized backlog

| Priority | Item | Rationale |
|---|---|---|
| **P1** | Measure on hardware: packet error rate vs attenuation, real throughput, video fps | Turn the theoretical budgets into actual success metrics |
| **P1** | Default to a compliant test configuration and document the cabled test setup | Remove regulatory risk for anyone reproducing the work |
| **P2** | Extract the shared modem into one module (it's currently copied in three scripts) | Lower maintenance cost; a fix lands everywhere at once |
| **P2** | Fix the missed control packets near the end of the receive buffer | Found by the test suite; improves stream start-up reliability |
| **P3** | UTF-8 text support | Enables non-Latin languages; low effort |
| **P3** | Encryption and authentication (e.g. AES-GCM) | Required before any real-world use; not needed for a lab prototype |

## Skills demonstrated

- **Scoping and incremental delivery:** three working releases, each a usable product.
- **Trade-off reasoning:** explicit choices between reliability, latency and complexity, with the cost of each stated.
- **Quantitative decision-making:** airtime budgets set the video settings and justified a protocol redesign.
- **Quality strategy:** a hardware-free test suite that removes a testing bottleneck and catches real defects.
- **Risk and compliance awareness:** regulatory and security constraints identified and planned for.
