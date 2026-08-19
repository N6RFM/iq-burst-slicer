# iq-burst-slicer

Shrink large IQ recordings that consist mostly of empty air with occasional
short bursts of signal — satellite beacons, pagers, key fobs/remotes, or any
other intermittent RF source — down to just the burst regions, while
preserving each burst's absolute UTC timestamp.

Typical result: **50-150x smaller**, losslessly (bursts are kept byte-for-byte
unmodified), for a signal with a low duty cycle.

Two output formats — [SigMF](https://sigmf.org) or [Digital RF](https://github.com/MITHaystack/digital_rf) — and tools to reconstruct a plain IQ file back from either.

| Tool | Purpose |
|---|---|
| `burst_slicer.py` | Detects bursts in a raw IQ file, discards everything else, writes SigMF or Digital RF output with per-burst absolute timestamps |
| `sigmf_to_iq.py` | Converts a sliced SigMF recording back into a plain raw IQ file, either compact (no gaps) or with the original timeline exactly reconstructed |
| `digitalrf_to_iq.py` | Same, for a sliced Digital RF recording |

No protocol-specific logic — detection is purely envelope-based, so this works on any bursty signal. See [Tuning](#tuning-for-your-signal) below.

## SigMF vs. Digital RF: full comparison

Both formats are legitimate, standard, widely-used choices for exactly this
situation — non-contiguous recording segments, each needing its own
absolute timestamp — so this isn't a "one is objectively better" choice.
We tested both directly rather than picking based on reputation; here's
the full picture, including where an earlier informal comparison of ours
was wrong (see the size row).

**What they have in common:** both are open, self-describing, non-proprietary
formats with real production use in the radio community; both support
storing multiple non-contiguous segments with per-segment absolute
timestamps (the exact property this tool depends on); both have Python
libraries and GNU Radio integration; neither requires you to invent your
own ad-hoc metadata scheme (e.g. encoding timestamps into filenames).

**Where they differ:**

| | SigMF | Digital RF |
|---|---|---|
| **Designed for** | Lightweight, general-purpose signal recordings of any size/duration | Continuous, long-duration, high-rate, often multi-channel scientific recording (built at MIT Haystack for ionospheric radar / space-weather work) |
| Dependencies | numpy only | `digital_rf` (pulls in `h5py`/HDF5) |
| Output structure | 2 plain files (`.sigmf-data` + human-readable JSON) | HDF5 directory/file hierarchy (chunked by time cadence) + our own JSON sidecar |
| Compression | None built in — would need an external step (`xz`/`zstd`) on top of our output if you want it | Built in — native gzip via `compression_level`, applied automatically as part of the write |
| Data integrity | Not implemented by this tool (SigMF spec supports an optional sha512 hash field, we don't set it) | Optional built-in HDF5 checksum (`checksum=True`), not enabled by default here |
| Size, no compression | 920,000 bytes | 1,824,415 bytes (~2x larger — HDF5 structural overhead dominates at this small, sparse scale) |
| Size, compression on | 920,000 bytes (nothing built in to enable) | 918,124 bytes (`compression_level>=1`) — **essentially tied, DRF slightly smaller** |
| Gap representation | Custom `captures` array (a SigMF-standard convention we rely on, not an inherent property of the format) | Native — gaps are a first-class part of the format's own sample indexing (`is_continuous=False`) |
| GNU Radio blocks | `blocks_sigmf_source/sink_minimal` — flagged **deprecated** in GNU Radio 3.10, doesn't even read sample rate from metadata | `gr_digital_rf` — actively maintained Sink/Source blocks |
| Tool ecosystem | [Inspectrum](https://github.com/miek/inspectrum) reads it natively | Haystack/openradar tooling (`drf_plot`, etc.); not read by Inspectrum |
| Readable without the format's library | Yes — `.sigmf-meta` is plain JSON, `.sigmf-data` is a flat binary `numpy.fromfile()` can read directly, zero special tooling | No — HDF5 tooling required either way, even just to inspect the file |
| Maturity / track record for this exact use case | Newer, simpler, DeepSig/GNU Radio community origin | Older, proven at large scale in production scientific recording (e.g. the HamSCI/Grape personal space-weather network) |

**The size result specifically is worth calling out**, because we initially
got it wrong: an earlier informal test compared Digital RF against a
mismatched, larger set of burst boundaries than what this tool actually
detects, making DRF look ~33% bigger than SigMF. Rerun properly — the
*exact same* burst samples fed to both formats — and with compression
enabled they're within 0.2% of each other. Size is **not** a real
differentiator here once compression is on; don't choose based on it.

**Bottom line:** if you want the simplest, dependency-free output that
works with Inspectrum and you can read with your eyes in a text editor —
use `--format sigmf` (the default). If you need Digital RF's ecosystem
(GNU Radio blocks, Haystack/openradar tooling), want built-in compression/
checksums without an extra step, or you're feeding into a pipeline that's
already built around continuous scientific-grade recording — use
`--format digitalrf` with compression enabled (the default,
`compression_level=9`). Either way, avoid Digital RF *without*
compression, which really is about 2x larger for this kind of sparse-burst
signal.

## Install

Base install (SigMF path only) needs just `numpy`:

```bash
pip install -r requirements.txt
```

For the Digital RF path, also install:

```bash
pip install -r requirements-digitalrf.txt
```

(`digital_rf` installed cleanly from a prebuilt wheel in our testing — no
system HDF5 headers needed, at least on Linux x86_64.)

## Usage

Want to try the tools immediately without your own data? The repo ships a
small example file — see [`examples/`](examples/).

`--utc-start` accepts the hyphen-separated, filename-safe time format used
by `gr-filerepeater_n6rfm` (`2026-08-18T17-48-10` — no colons), and also
still accepts standard ISO8601 with colons (`2026-08-18T17:48:10`) for
backward compatibility. Either way, the UTC start time is automatically
appended to the output filename as `_{timestamp}_utc` (normalized to the
hyphenated form), so `-o my_capture_sliced` with `--utc-start
2026-08-18T17-48-10` writes `my_capture_sliced_2026-08-18T17-48-10_utc.*`
rather than plain `my_capture_sliced.*` — the timestamp is always visible
in the filename itself, not just inside the metadata.

**If your input filename already carries a local timestamp** (the
`gr-filerepeater_n6rfm` convention, e.g.
`DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq`), use
`--utc-offset` instead of `--utc-start` — give the UTC offset in hours
(e.g. `-4` for Rhode Island EDT) and the tool reads the embedded local
time from the filename and converts it for you:

```bash
python3 burst_slicer.py DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq \
    --fs 50000 --utc-offset -4 -o sliced_output
#   13:30 local (embedded in filename) - (-4h) = 17:30 UTC
#   -> sliced_output_2026-08-18T17-30-00_utc.sigmf-data / .sigmf-meta
```

`--utc-start` and `--utc-offset` are mutually exclusive if both are given —
but **both are optional**. If neither is given, the timestamp already
embedded in the input filename is reused **exactly as written, with no
timezone assumption** — it is *not* treated as UTC, just passed through
unchanged, since we weren't told what timezone it's actually in:

```bash
python3 burst_slicer.py DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq --fs 50000
#   T13-30-00 is reused exactly as-is -- no UTC claim, no conversion,
#   -> DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.sigmf-data / .sigmf-meta
#      (same filename, just the .sigmf-data/.sigmf-meta suffix added -- no "_utc" marker,
#       since nothing was actually verified as UTC)
```

If the input filename has no recognizable embedded timestamp and neither
`--utc-start` nor `--utc-offset` was given, the tool fails with a clear
error rather than guessing.

**`-o`/`--output` is optional.** If omitted, the input filename is reused
automatically:

- **If the UTC time was verified** (`--utc-start` or `--utc-offset` was
  given) and the input filename already embeds a `gr-filerepeater_n6rfm`
  -style timestamp, that timestamp is **replaced** with the corrected UTC
  time (not appended a second time) — so
  `DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq` with
  `--utc-offset -4` and no `-o` writes
  `DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T17-30-00_utc.sigmf-data`
  / `.sigmf-meta` — same filename, local time swapped for UTC, and a
  `_utc` marker added since that conversion was actually verified.
- **If the UTC time was verified** but there's no embedded timestamp to
  replace, the usual `_{timestamp}_utc` suffix is appended to the input
  filename's stem instead, same as when `-o` is given explicitly.
- **If the UTC time was *not* verified** (neither flag given), the input
  filename's stem is reused **completely unchanged** — no substitution,
  no `_utc` suffix, since nothing was actually confirmed as UTC. This
  applies whether or not `-o` was given: an explicit `-o` name is also
  used exactly as given, with no `_utc`-claiming timestamp appended.

`--format` is also optional and defaults to `sigmf`, so the shortest
possible invocation — input filename already has a usable timestamp, no
`-o`, no `--format` — is just:

```bash
python3 burst_slicer.py DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq \
    --fs 50000 --utc-offset -4
```

```bash
# Slice bursts out of a large recording (SigMF, the default)
python3 burst_slicer.py my_capture.iq --fs 50000 \
    --utc-start 2026-08-18T17-48-10 \
    -o my_capture_sliced

#   -> my_capture_sliced_2026-08-18T17-48-10_utc.sigmf-data   (raw samples, bursts only)
#   -> my_capture_sliced_2026-08-18T17-48-10_utc.sigmf-meta   (JSON, one timestamped entry per burst)

# ...or Digital RF instead:
python3 burst_slicer.py my_capture.iq --fs 50000 \
    --utc-start 2026-08-18T17-48-10 \
    -o my_capture_sliced_drf --format digitalrf

#   -> my_capture_sliced_drf_2026-08-18T17-48-10_utc/ch0/*.h5          (HDF5 data, gaps native to the format)
#   -> my_capture_sliced_drf_2026-08-18T17-48-10_utc/burstslicer_info.json  (JSON sidecar)

# Reconstruct a plain IQ file if you need one, with the ORIGINAL timeline
# exactly restored (verified byte-identical burst content, exact original
# file length):
python3 sigmf_to_iq.py my_capture_sliced_2026-08-18T17-48-10_utc.sigmf-meta \
    -o my_capture_restored.iq --full-timeline --noise-fill
# or, from Digital RF:
python3 digitalrf_to_iq.py my_capture_sliced_drf_2026-08-18T17-48-10_utc \
    -o my_capture_restored.iq --full-timeline --noise-fill

# Or just the bursts concatenated with no gaps (smallest possible plain IQ):
python3 sigmf_to_iq.py my_capture_sliced_2026-08-18T17-48-10_utc.sigmf-meta \
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
    --utc-start 2026-08-18T12-00-00 -o sliced \
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
pip install -r requirements-dev.txt
pytest tests/
```

The test suite generates a synthetic bursty signal (deliberately different
from any specific real-world protocol: different modulation, sample rate,
burst duration, and irregular spacing including two bursts close enough
together to exercise the merge-vs-split logic) and verifies the full
slice → reconstruct round trip reproduces the original file exactly:
same length, same burst positions, byte-identical burst content —
for both SigMF and Digital RF. Digital RF tests are skipped automatically
if `digital_rf` isn't installed.

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
