#!/usr/bin/env python3
"""
burst_slicer.py

General-purpose tool for slicing the signal-bearing regions out of a large
IQ recording that consists mostly of empty/noise-only time with occasional
short bursts of signal (e.g. satellite beacons, pagers, remote controls,
vehicle key fobs, any bursty/intermittent RF source) -- discarding the
(often 95%+) dead air between bursts, and writing the result as a SigMF
recording pair:

    OUTPUT.sigmf-data   -- raw complex samples, bursts concatenated back-to-back
    OUTPUT.sigmf-meta   -- JSON metadata, with one "capture" segment per burst
                            recording that burst's *absolute UTC start time*
                            and its sample offset within the sliced data file

This uses SigMF's standard multi-capture-segment mechanism, which is
specifically designed for exactly this situation: non-contiguous chunks of a
real recording, concatenated into one file, each needing its own timestamp.
See https://sigmf.org

Works on any bursty signal; detection is purely envelope-based (no
protocol-specific assumptions). Typical size reduction depends on your
signal's duty cycle -- for something like a satellite beacon transmitting a
fraction-of-a-second burst once a minute, expect 50-150x.

Usage:
    python3 burst_slicer.py INPUT.iq --fs 50000 --utc-start 2026-08-18T17:48:10 -o sliced_output
    python3 burst_slicer.py INPUT.iq --fs 50000 --utc-start 2026-08-18T17:48:10 -o sliced_output --thresh-factor 1.5

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

Requires: numpy
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='Path to raw complex-float32 IQ file')
    ap.add_argument('--fs', type=float, required=True, help='Sample rate in Hz')
    ap.add_argument('--utc-start', required=True,
                     help='ISO8601 UTC timestamp of sample 0 in the INPUT file, '
                          'e.g. 2026-08-18T17:48:10')
    ap.add_argument('-o', '--output', required=True,
                     help='Output basename (without extension) -- writes '
                          'OUTPUT.sigmf-data and OUTPUT.sigmf-meta')
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
    ap.add_argument('--author', default='', help='Optional core:author for the SigMF metadata')
    ap.add_argument('--description', default='Burst-sliced IQ recording',
                     help='core:description for the SigMF metadata')
    args = ap.parse_args()

    fs = args.fs
    t_start = dt.datetime.fromisoformat(args.utc_start)
    if t_start.tzinfo is None:
        t_start = t_start.replace(tzinfo=dt.timezone.utc)

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

    out_base = Path(args.output)
    data_path = out_base.with_suffix('.sigmf-data')
    meta_path = out_base.with_suffix('.sigmf-meta')

    captures = []
    sample_cursor = 0
    with open(data_path, 'wb') as fout:
        for i, (lo, hi) in enumerate(slices):
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
            'core:description': args.description,
            'core:author': args.author,
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

    in_size = in_path.stat().st_size
    out_size = data_path.stat().st_size
    print(f'\nWrote {data_path}  ({out_size} bytes = {out_size/1e6:.3f} MB)')
    print(f'Wrote {meta_path}  ({len(captures)} capture segments)')
    print(f'\nSize: {in_size/1e6:.2f} MB -> {out_size/1e6:.3f} MB '
          f'({in_size/out_size:.1f}x smaller)')


if __name__ == '__main__':
    main()
