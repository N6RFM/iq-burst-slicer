#!/usr/bin/env python3
"""
digitalrf_to_iq.py

Converts a sliced Digital RF recording (as produced by burst_slicer.py
--format digitalrf) back into a plain raw complex-float32 IQ file.

Two modes, identical in behavior to sigmf_to_iq.py:

  --compact (default)
      Just concatenates the burst samples, no gaps. Smallest output.

  --full-timeline
      Reconstructs a file with the SAME overall duration and burst spacing
      as the original recording (using the original recording bounds
      recorded in the burstslicer_info.json sidecar). Filler defaults to
      zeros; use --noise-fill if you intend to re-run a demodulator on the
      output -- see sigmf_to_iq.py's module docstring for why plain
      zero-fill corrupts FM-discriminator-based decoding at burst edges
      (the same applies here, this is a property of the reconstruction,
      not the source format).

Usage:
    python3 digitalrf_to_iq.py sliced_drf_dir -o restored.iq --compact
    python3 digitalrf_to_iq.py sliced_drf_dir -o restored_full.iq --full-timeline --noise-fill

Requires: numpy, digital_rf (pip install digital_rf)
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from _reconstruct import reconstruct_compact, reconstruct_full_timeline


def load_digitalrf(drf_dir):
    try:
        import digital_rf as drf
    except ImportError:
        raise SystemExit(
            "Error: digitalrf_to_iq.py requires the 'digital_rf' package.\n"
            "Install it with: pip install digital_rf")

    drf_dir = Path(drf_dir)
    info_path = drf_dir / 'burstslicer_info.json'
    if not info_path.exists():
        raise SystemExit(f"Error: {info_path} not found -- is this a directory "
                          f"written by burst_slicer.py --format digitalrf?")
    with open(info_path) as f:
        info = json.load(f)

    fs = float(info['sample_rate'])
    reader = drf.DigitalRFReader(str(drf_dir))
    channels = reader.get_channels()
    if not channels:
        raise SystemExit(f"Error: no channels found in {drf_dir}")
    channel = channels[0]
    bounds_start, bounds_end = reader.get_bounds(channel)

    segments = []
    for cap in info['captures']:
        s0_global = int(cap['sample_start']) + bounds_start_offset(info, fs)
        n = int(cap['sample_count'])
        t_abs = dt.datetime.fromisoformat(cap['datetime'].replace('Z', '+00:00'))
        chunk = reader.read_vector(s0_global, n, channel)
        segments.append((chunk, t_abs))

    orig_start = dt.datetime.fromisoformat(info['original_utc_start'].replace('Z', '+00:00'))
    orig_total = info.get('original_sample_count')

    return segments, fs, orig_start, orig_total


def bounds_start_offset(info, fs):
    """The DigitalRFWriter's start_global_index was computed from
    original_utc_start (see burst_slicer.py) -- reproduce that here so
    reader.read_vector() global indices match what was written."""
    t0 = dt.datetime.fromisoformat(info['original_utc_start'].replace('Z', '+00:00'))
    return int(t0.timestamp() * fs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='Path to the Digital RF directory to restore '
                                   '(the one containing burstslicer_info.json and ch0/)')
    ap.add_argument('-o', '--output', required=True, help='Output raw IQ file path')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--compact', action='store_true', default=True,
                       help='(default) Concatenate bursts with no gaps -- smallest output')
    mode.add_argument('--full-timeline', action='store_true',
                       help='Reconstruct full original duration/spacing')
    ap.add_argument('--noise-fill', action='store_true',
                     help='With --full-timeline, fill gaps with synthetic noise instead of '
                          'zeros. RECOMMENDED if the output will be re-decoded -- see module '
                          'docstring.')
    args = ap.parse_args()

    raw_segments, fs, orig_start, orig_total = load_digitalrf(args.input)
    print(f'Loaded {args.input}: {len(raw_segments)} burst segment(s), fs={fs:.0f} Hz')

    # Adapt to the shared _reconstruct interface, which expects (iq_array,
    # segments=[(s0,s1,t_abs),...]) against one shared array. Since
    # DigitalRFReader gives us each chunk independently, concatenate them
    # into one local array and build matching (s0,s1) offsets into it.
    iq_chunks = [chunk for chunk, _ in raw_segments]
    iq = np.concatenate(iq_chunks) if iq_chunks else np.array([], dtype=np.complex64)
    segments = []
    cursor = 0
    for chunk, t_abs in raw_segments:
        segments.append((cursor, cursor + len(chunk), t_abs))
        cursor += len(chunk)

    out_path = Path(args.output)

    if args.full_timeline:
        print(f'  using recorded original start time: {orig_start.isoformat()}'
              + (f', original length {orig_total} samples' if orig_total else ''))
        written, _ = reconstruct_full_timeline(
            iq, fs, segments, out_path, orig_start=orig_start, orig_total=orig_total,
            noise_fill=args.noise_fill)
        print(f'\nWrote {out_path}  ({written} samples, {written/fs:.2f}s, '
              f'{out_path.stat().st_size} bytes)')
        print(f'Reconstructed timeline spans {written/fs:.2f}s '
              f'(gaps filled with {"synthetic noise" if args.noise_fill else "zeros"})')
    else:
        written = reconstruct_compact(iq, segments, out_path)
        print(f'\nWrote {out_path}  ({written} samples, {written/fs:.2f}s, '
              f'{out_path.stat().st_size} bytes)  [compact -- no gaps, bursts back-to-back]')


if __name__ == '__main__':
    main()
