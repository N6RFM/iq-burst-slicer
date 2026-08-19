#!/usr/bin/env python3
"""
burst_slicer.py

General-purpose tool for slicing the signal-bearing regions out of a large
IQ recording that consists mostly of empty/noise-only time with occasional
short bursts of signal (e.g. satellite beacons, pagers, remote controls,
vehicle key fobs, any bursty/intermittent RF source) -- discarding the
(often 95%+) dead air between bursts.

Two output formats, chosen with --format:

  sigmf (default)
      OUTPUT.sigmf-data   -- raw complex samples, bursts concatenated back-to-back
      OUTPUT.sigmf-meta   -- JSON metadata, with one "capture" segment per burst
                              recording that burst's *absolute UTC start time*
                              and its sample offset within the sliced data file
      Uses SigMF's standard multi-capture-segment mechanism. See https://sigmf.org
      Simple, dependency-free (just numpy), human-readable metadata, and
      (per our own testing) smaller output for this sparse-burst use case
      than Digital RF's HDF5 structure -- see README for real numbers.

  digitalrf
      OUTPUT/ch0/*.h5           -- Digital RF HDF5 data, written with
                                    is_continuous=False so gaps between
                                    bursts are natively represented by the
                                    format's own sample indexing (no custom
                                    metadata scheme needed for that part)
      OUTPUT/burstslicer_info.json -- small sidecar with the same original-
                                    recording bounds info as the SigMF path,
                                    plus the per-burst capture list, for
                                    parity with the sigmf output
      Requires the `digital_rf` package (pip install digital_rf). Has
      native gzip compression (--drf-compression-level, default 9) and
      works with Digital RF's own mature GNU Radio blocks (gr_digital_rf)
      and the wider Haystack/openradar tooling ecosystem.

      Size, tested on a real 4-burst satellite recording (same exact burst
      samples fed to both formats for a fair comparison): with compression
      enabled (any level >=1), DRF is essentially tied with plain SigMF
      (918KB vs 920KB -- DRF slightly smaller). With compression_level=0,
      DRF is roughly 2x larger than SigMF due to HDF5 structural overhead.
      So: use SigMF if you want the simplest possible dependency-free
      output; use --format digitalrf with the default compression if you
      need Digital RF's ecosystem (GNU Radio blocks, Haystack tooling,
      native gap-aware indexing) -- you won't pay a size penalty for it.

Works on any bursty signal; detection is purely envelope-based (no
protocol-specific assumptions). Typical size reduction depends on your
signal's duty cycle -- for something like a satellite beacon transmitting a
fraction-of-a-second burst once a minute, expect 50-150x versus the original
file, before even considering which output format to use.

Usage:
    python3 burst_slicer.py INPUT.iq --fs 50000 --utc-start 2026-08-18T17-48-10 -o sliced_output
    python3 burst_slicer.py INPUT.iq --fs 50000 --utc-start 2026-08-18T17-48-10 -o sliced_output --thresh-factor 1.5
    python3 burst_slicer.py INPUT.iq --fs 50000 --utc-start 2026-08-18T17-48-10 -o sliced_drf --format digitalrf

    # Or, if the input filename already carries a local timestamp in the
    # gr-filerepeater_n6rfm convention (YYYY_MM_DD_THH-MM-SS), skip
    # --utc-start and just give the UTC offset instead:
    python3 burst_slicer.py DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq \
        --fs 50000 --utc-offset -4 -o sliced_output

Tuning for your signal:
    --thresh-factor   how far above the noise floor (median envelope) a
                       region must be to count as a burst. Lower this for
                       weak/marginal signals; raise it if noise spikes are
                       being falsely detected as bursts.
    --block-s         time resolution of the coarse envelope detector.
                       Should be well under your shortest burst's duration
                       (default 0.05s suits sub-second bursts; use a smaller
                       value for very short bursts, e.g. microsecond-scale
                       pulses, or larger for multi-second bursts).
    --min-gap-s       minimum silence gap required to treat two nearby
                       bursts as separate (rather than merging them into one
                       region). Set this shorter than your signal's true
                       burst-to-burst spacing but longer than any pause
                       *within* a single burst.

Requires: numpy (plus digital_rf, only if using --format digitalrf)
"""

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np


def parse_utc_arg(s):
    """Parse a UTC timestamp argument. Accepts the hyphen-separated time
    format used by gr-filerepeater_n6rfm (2026-08-18T17-48-10, filename-
    safe -- no colons) as well as standard ISO8601 (2026-08-18T17:48:10),
    so existing scripts/muscle memory using either format keep working."""
    if 'T' in s:
        date_part, time_part = s.split('T', 1)
        time_part = time_part.replace('-', ':')
        s = f'{date_part}T{time_part}'
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t


_FILENAME_TIMESTAMP_RE = re.compile(
    r'(\d{4})_(\d{2})_(\d{2})_T(\d{2})-(\d{2})-(\d{2})')


def parse_local_start_from_filename(path, utc_offset_hours):
    """Extract the LOCAL recording-start timestamp embedded in a
    gr-filerepeater_n6rfm-style input filename (e.g.
    DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq embeds
    2026-08-18 13:30:00 local) and convert it to UTC using the given
    offset in hours (e.g. -4 for Rhode Island EDT, so that
    UTC = local - utc_offset_hours). Raises SystemExit with a clear
    message if the filename doesn't contain a recognizable timestamp."""
    m = _FILENAME_TIMESTAMP_RE.search(Path(path).name)
    if not m:
        raise SystemExit(
            f"Error: --utc-offset was given, but no embedded timestamp of the form "
            f"YYYY_MM_DD_THH-MM-SS (gr-filerepeater_n6rfm convention) was found in the "
            f"input filename '{Path(path).name}'. Use --utc-start instead to specify the "
            f"UTC start time directly.")
    year, month, day, hour, minute, second = (int(x) for x in m.groups())
    local_dt = dt.datetime(year, month, day, hour, minute, second)
    utc_dt = local_dt - dt.timedelta(hours=utc_offset_hours)
    return utc_dt.replace(tzinfo=dt.timezone.utc)


def find_bursts(iq, fs, block_s=0.05, thresh_factor=1.8, min_gap_s=1.0):
    """Coarse envelope-based burst region detection (block-averaged, robust
    to per-sample noise spikes)."""
    block = max(1, int(block_s * fs))
    n_blocks = len(iq) // block
    if n_blocks == 0:
        return []
    mag = np.abs(iq[:n_blocks * block]).reshape(n_blocks, block)
    env = mag.mean(axis=1)
    thresh = np.median(env) * thresh_factor
    above = env > thresh
    idx = np.where(above)[0]
    if len(idx) == 0:
        return []
    runs = []
    start = idx[0]
    prev = idx[0]
    min_gap_blocks = max(1, int(min_gap_s / block_s))
    for i in idx[1:]:
        if i > prev + min_gap_blocks:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    pad = block * 2
    return [(max(0, s * block - pad), min(len(iq), (e + 1) * block + pad)) for s, e in runs]


def write_sigmf(iq, fs, t_start, slices, out_base, author, description):
    data_path = out_base.with_suffix('.sigmf-data')
    meta_path = out_base.with_suffix('.sigmf-meta')

    captures = []
    sample_cursor = 0
    with open(data_path, 'wb') as fout:
        for lo, hi in slices:
            chunk = iq[lo:hi]
            chunk.astype(np.complex64).tofile(fout)
            t_abs = t_start + dt.timedelta(seconds=lo / fs)
            captures.append({
                'core:sample_start': sample_cursor,
                'core:datetime': t_abs.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'core:sample_count': len(chunk),
            })
            sample_cursor += len(chunk)

    meta = {
        'global': {
            'core:datatype': 'cf32_le',
            'core:sample_rate': fs,
            'core:version': '1.0.0',
            'core:description': description,
            'core:author': author,
            'core:recorder': 'burst_slicer.py',
            'core:extensions': [],
            # non-core custom fields: the ORIGINAL (pre-slicing) recording's
            # start time and total sample count, so a full-timeline
            # reconstruction can exactly reproduce the original file's
            # duration (including any lead-in/trailing silence), not just
            # the span from first burst to last burst.
            'burstslicer:original_utc_start': t_start.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'burstslicer:original_sample_count': len(iq),
        },
        'captures': captures,
        'annotations': [],
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    out_size = data_path.stat().st_size
    print(f'\nWrote {data_path}  ({out_size} bytes = {out_size/1e6:.3f} MB)')
    print(f'Wrote {meta_path}  ({len(captures)} capture segments)')
    return out_size


def write_digitalrf(iq, fs, t_start, slices, out_base, author, description,
                     compression_level):
    try:
        import digital_rf as drf
    except ImportError:
        raise SystemExit(
            "Error: --format digitalrf requires the 'digital_rf' package.\n"
            "Install it with: pip install digital_rf")

    out_dir = Path(out_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    channel_dir = out_dir / 'ch0'
    channel_dir.mkdir(exist_ok=True)

    start_global_index = int(t_start.timestamp() * fs)

    writer = drf.DigitalRFWriter(
        str(channel_dir), dtype=np.complex64,
        subdir_cadence_secs=3600, file_cadence_millisecs=1000,
        start_global_index=start_global_index,
        sample_rate_numerator=int(fs), sample_rate_denominator=1,
        compression_level=compression_level, checksum=False,
        is_complex=True, num_subchannels=1, is_continuous=False,
        marching_periods=False,
    )

    captures = []
    for lo, hi in slices:
        lo = int(lo)
        chunk = iq[lo:hi]
        writer.rf_write(chunk, next_sample=lo)
        t_abs = t_start + dt.timedelta(seconds=lo / fs)
        captures.append({
            'sample_start': lo,
            'datetime': t_abs.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'sample_count': len(chunk),
        })
    writer.close()

    info = {
        'sample_rate': fs,
        'description': description,
        'author': author,
        'recorder': 'burst_slicer.py',
        'original_utc_start': t_start.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        'original_sample_count': len(iq),
        'captures': captures,
    }
    info_path = out_dir / 'burstslicer_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    total = 0
    import os
    for root, _, files in os.walk(out_dir):
        for fn in files:
            total += (Path(root) / fn).stat().st_size

    print(f'\nWrote {channel_dir}/ ({len(captures)} burst(s) written, gaps native to the format)')
    print(f'Wrote {info_path}')
    print(f'Total on-disk size: {total} bytes = {total/1e6:.3f} MB')
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='Path to raw complex-float32 IQ file')
    ap.add_argument('--fs', type=float, required=True, help='Sample rate in Hz')
    utc_group = ap.add_mutually_exclusive_group(required=True)
    utc_group.add_argument('--utc-start',
                     help='UTC timestamp of sample 0 in the INPUT file, in the hyphen-separated '
                          'format used by gr-filerepeater_n6rfm, e.g. 2026-08-18T17-48-10 '
                          '(standard ISO8601 with colons, e.g. 2026-08-18T17:48:10, is also '
                          'still accepted)')
    utc_group.add_argument('--utc-offset', type=float,
                     help='Alternative to --utc-start: instead of specifying UTC directly, '
                          'read the LOCAL recording-start timestamp already embedded in the '
                          'gr-filerepeater_n6rfm-style input filename (e.g. '
                          'DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq '
                          'embeds 2026-08-18 13:30:00 local) and convert it to UTC using this '
                          'offset in hours, e.g. -4 for Rhode Island EDT '
                          '(UTC = local time - offset).')
    ap.add_argument('-o', '--output', required=True,
                     help='Output basename/directory (without extension) -- for '
                          '--format sigmf writes OUTPUT.sigmf-data/.sigmf-meta; '
                          'for --format digitalrf writes an OUTPUT/ directory')
    ap.add_argument('--format', choices=['sigmf', 'digitalrf'], default='sigmf',
                     help="Output format (default 'sigmf'). See this script's "
                          "module docstring for a real size comparison between "
                          "the two -- they are NOT equivalent for this use case.")
    ap.add_argument('--drf-compression-level', type=int, default=9, choices=range(10),
                     help='(--format digitalrf only) gzip compression level 0-9 '
                          '(default 9, max compression)')
    ap.add_argument('--thresh-factor', type=float, default=1.8,
                     help='Envelope threshold = median * this factor (default 1.8; '
                          'lower to catch weaker bursts)')
    ap.add_argument('--block-s', type=float, default=0.05,
                     help='Coarse envelope-detector time resolution in seconds (default 0.05). '
                          'Should be well under your shortest burst duration.')
    ap.add_argument('--min-gap-s', type=float, default=1.0,
                     help='Minimum silence gap (seconds) to treat two nearby bursts as separate '
                          'rather than merging into one region (default 1.0). Set shorter than '
                          'your true burst-to-burst spacing but longer than any pause within a '
                          'single burst.')
    ap.add_argument('--pad-s', type=float, default=0.0,
                     help='Extra seconds of padding to keep on each side of every '
                          'detected burst region, beyond the built-in margin (default 0)')
    ap.add_argument('--author', default='', help='Optional author field for the metadata')
    ap.add_argument('--description', default='Burst-sliced IQ recording',
                     help='Optional description field for the metadata')
    args = ap.parse_args()

    fs = args.fs
    if args.utc_start is not None:
        t_start = parse_utc_arg(args.utc_start)
    else:
        t_start = parse_local_start_from_filename(args.input, args.utc_offset)
        local_preview = t_start + dt.timedelta(hours=args.utc_offset)
        print(f'Parsed local start time from filename: {local_preview.strftime("%Y-%m-%d %H:%M:%S")} '
              f'(offset {args.utc_offset:+.1f}h) -> UTC {t_start.strftime("%Y-%m-%dT%H:%M:%S")}Z')

    in_path = Path(args.input)
    raw = np.fromfile(in_path, dtype=np.float32)
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    iq = raw.view(np.complex64)
    print(f'Input: {in_path}  ({len(iq)} samples, {len(iq)/fs:.2f}s, '
          f'{in_path.stat().st_size/1e6:.2f} MB)')

    bursts = find_bursts(iq, fs, block_s=args.block_s, thresh_factor=args.thresh_factor,
                          min_gap_s=args.min_gap_s)
    print(f'{len(bursts)} burst region(s) detected')

    pad = int(args.pad_s * fs)
    slices = []
    for s0, s1 in bursts:
        lo = max(0, s0 - pad)
        hi = min(len(iq), s1 + pad)
        slices.append((lo, hi))

    in_size = in_path.stat().st_size
    # Append the UTC start time to the output basename, in the same
    # hyphen-separated, filename-safe format used by gr-filerepeater_n6rfm,
    # regardless of which format --utc-start was given in -- e.g.
    # -o sliced --utc-start 2026-08-18T12-00-00 writes
    # sliced_2026-08-18T12-00-00_utc.sigmf-data / .sigmf-meta
    utc_suffix = t_start.strftime('%Y-%m-%dT%H-%M-%S')
    out_base = Path(f'{args.output}_{utc_suffix}_utc')

    if args.format == 'sigmf':
        out_size = write_sigmf(iq, fs, t_start, slices, out_base, args.author, args.description)
    else:
        out_size = write_digitalrf(iq, fs, t_start, slices, out_base, args.author,
                                    args.description, args.drf_compression_level)

    print(f'\nSize: {in_size/1e6:.2f} MB -> {out_size/1e6:.3f} MB '
          f'({in_size/out_size:.1f}x smaller)')


if __name__ == '__main__':
    main()
