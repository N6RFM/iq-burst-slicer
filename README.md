# iq-burst-slicer

Shrink large IQ recordings that consist mostly of empty air with occasional
short bursts of signal — satellite beacons, pagers, key fobs/remotes, or any
other intermittent RF source — down to just the burst regions, while
preserving each burst's absolute UTC timestamp.

Typical result: **50-150x smaller**, losslessly (bursts are kept byte-for-byte
unmodified), for a signal with a low duty cycle.

Two tools:

| Tool | Purpose |
|---|---|
| `burst_slicer.py` | Detects bursts in a raw IQ file, discards everything else, writes a [SigMF](https://sigmf.org) recording pair with one timestamped capture segment per burst |
| `sigmf_to_iq.py` | Converts a sliced SigMF recording back into a plain raw IQ file, either compact (no gaps) or with the original timeline exactly reconstructed |

No protocol-specific logic — detection is purely envelope-based, so this works on any bursty signal. See [Tuning](#tuning-for-your-signal) below.

## Why SigMF?

[SigMF](https://sigmf.org)'s `captures` array is designed for exactly this
situation: non-contiguous chunks of a real recording, concatenated into one
data file, each needing its own absolute timestamp. Using a standard format
(rather than encoding timestamps into filenames, which is fragile — ask us
how we know) means the output is readable by any SigMF-aware tool
(e.g. [Inspectrum](https://github.com/miek/inspectrum)), and the metadata
stays a plain, human-readable JSON file alongside the data.

## Install

Only dependency is `numpy`:

```bash
pip install numpy
```

Or with the repo's pinned requirements:

```bash
pip install -r requirements.txt
```

## Usage

Want to try the tools immediately without your own data? The repo ships a
small example file — see [`examples/`](examples/).

```bash
# Slice bursts out of a large recording
python3 burst_slicer.py my_capture.iq --fs 50000 \
    --utc-start 2026-08-18T17:48:10 \
    -o my_capture_sliced

#   -> my_capture_sliced.sigmf-data   (raw samples, bursts only)
#   -> my_capture_sliced.sigmf-meta   (JSON, one timestamped entry per burst)

# Reconstruct a plain IQ file if you need one, with the ORIGINAL timeline
# exactly restored (verified byte-identical burst content, exact original
# file length):
python3 sigmf_to_iq.py my_capture_sliced.sigmf-meta \
    -o my_capture_restored.iq --full-timeline --noise-fill

# Or just the bursts concatenated with no gaps (smallest possible plain IQ):
python3 sigmf_to_iq.py my_capture_sliced.sigmf-meta \
    -o my_capture_compact.iq --compact
```

### `--full-timeline` fill modes

- **`--noise-fill`** (recommended if you'll re-run any kind of demodulator
  on the output): fills gaps with synthetic low-level Gaussian noise matched
  to the recording's own noise floor.
- **plain zero-fill** (default without `--noise-fill`): true silence in the
  gaps. Looks fine for visual inspection, but **will corrupt FM-discriminator-
  based demodulation right at the start of the following burst** — dividing
  by a near-zero magnitude in the phase calculation produces large spurious
  phase transients at the zero-to-signal boundary. Confirmed in testing: this
  dropped a working decoder's CRC-verified block count to zero. Use
  `--noise-fill` instead if the output needs to remain decodable.

## Tuning for your signal

```bash
python3 burst_slicer.py other_signal.iq --fs 200000 \
    --utc-start 2026-08-18T12:00:00 -o sliced \
    --thresh-factor 1.5 --block-s 0.01 --min-gap-s 0.5
```

- **`--thresh-factor`** (default `1.8`) — sensitivity above the noise floor.
  Lower for weak/marginal signals; raise if noise spikes are being falsely
  detected as bursts.
- **`--block-s`** (default `0.05`) — time resolution of detection. Must be
  well under your shortest burst's duration.
- **`--min-gap-s`** (default `1.0`) — minimum silence gap to treat two
  nearby bursts as separate rather than merging into one region.

If unsure, start with the defaults, check the printed burst count/durations,
and adjust — or look at a spectrogram first to see your actual burst
length/spacing before picking values.

## Testing

```bash
pip install pytest
pytest tests/
```

The test suite generates a synthetic bursty signal (deliberately different
from any specific real-world protocol: different modulation, sample rate,
burst duration, and irregular spacing including two bursts close enough
together to exercise the merge-vs-split logic) and verifies the full
slice → reconstruct round trip reproduces the original file exactly:
same length, same burst positions, byte-identical burst content.

## Limitations

- `--compact` output packs bursts back-to-back with no gaps; if you feed
  that into something that re-detects bursts by looking for silence gaps,
  bursts originally spaced far apart may get merged into one detected
  region. Use SigMF-aware tooling (reading the `captures` array directly)
  or `--full-timeline` reconstruction if that matters for your downstream
  use.
- Detection is envelope-based only; very low-SNR bursts may need
  `--thresh-factor` tuned down, and extremely low-SNR bursts may not be
  reliably separable from noise at all.

## License

MIT — see [LICENSE](LICENSE).
