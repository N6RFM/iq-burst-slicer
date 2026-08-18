#!/usr/bin/env python3
"""
sigmf_to_iq.py

Converts a sliced SigMF recording (as produced by burst_slicer.py --format
sigmf) back into a plain raw complex-float32 IQ file.

Two modes:

  --compact (default)
      Just concatenates the burst samples as stored in the .sigmf-data file,
      with no gaps. Smallest output, but bursts will appear back-to-back
      rather than at their real original spacing.

  --full-timeline
      Reconstructs a file with the SAME overall duration and burst spacing
      as the original recording, by inserting the correct number of filler
      samples between capture segments (computed from each segment's
      recorded core:datetime, and from the original recording's start time/
      length if burst_slicer.py recorded them). Useful for tools that
      expect one continuous, timeline-accurate recording (e.g. Inspectrum,
      or comparing against another simultaneous capture).

      Filler defaults to zeros (true silence). IMPORTANT: zero-fill is fine
      for visual/spectrogram inspection, but will corrupt FM-discriminator-
      based demodulation right at the start of each burst -- dividing by a
      near-zero magnitude in the phase calculation produces large spurious
      phase transients at the zero-to-signal boundary, which bleed into the
      following burst through the lowpass filter and can corrupt the frame-
      sync region specifically (confirmed: this reduced decode results from
      0/6,6/6,6/6,1/6 to 0/6,0/6,0/6,0/6 blocks CRC-verified on a test file).
      Use --noise-fill instead if you intend to re-run a demodulator on the
      reconstructed file.

      --noise-fill fills gaps with synthetic low-level Gaussian noise
      matched to the recording's own noise floor instead of zeros, which
      avoids the above problem (confirmed: restores 6/6,6/6 CRC-verified
      decode on the same test file). This is SYNTHETIC data, not real
      captured samples -- it makes the reconstructed file behave correctly
      for demodulation and look natural in a spectrogram, but the gap
      regions are not what was actually in the air at that time.

Usage:
    python3 sigmf_to_iq.py sliced.sigmf-meta -o restored.iq --compact
    python3 sigmf_to_iq.py sliced.sigmf-meta -o restored_full.iq --full-timeline
    python3 sigmf_to_iq.py sliced.sigmf-meta -o restored_full.iq --full-timeline --noise-fill

Requires: numpy
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from _reconstruct import reconstruct_compact, reconstruct_full_timeline


def load_sigmf(meta_path):
    meta_path = Path(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    fs = float(meta['global']['core:sample_rate'])
    data_path = meta_path.with_suffix('.sigmf-data')
    raw = np.fromfile(data_path, dtype=np.float32)
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    iq = raw.view(np.complex64)

    captures = meta.get('captures', [])
    segments = []
    for i, cap in enumerate(captures):
        s0 = int(cap['core:sample_start'])
        if i + 1 < len(captures):
            s1 = int(captures[i + 1]['core:sample_start'])
        elif 'core:sample_count' in cap:
            s1 = s0 + int(cap['core:sample_count'])
        else:
            s1 = len(iq)
        t_abs = None
        if 'core:datetime' in cap:
            ts = cap['core:datetime'].replace('Z', '+00:00')
            t_abs = dt.datetime.fromisoformat(ts)
        segments.append((s0, s1, t_abs))

    orig_start = None
    orig_start_str = meta['global'].get('burstslicer:original_utc_start')
    if orig_start_str:
        orig_start = dt.datetime.fromisoformat(orig_start_str.replace('Z', '+00:00'))
    orig_total = meta['global'].get('burstslicer:original_sample_count')

    return iq, fs, segments, orig_start, orig_total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='Path to the .sigmf-meta (or .sigmf-data) file to restore')
    ap.add_argument('-o', '--output', required=True, help='Output raw IQ file path')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--compact', action='store_true', default=True,
                       help='(default) Concatenate bursts with no gaps -- smallest output')
    mode.add_argument('--full-timeline', action='store_true',
                       help='Reconstruct full original duration/spacing by inserting '
                            'filler samples between bursts, sized from each capture\'s '
                            'recorded core:datetime')
    ap.add_argument('--noise-fill', action='store_true',
                     help='With --full-timeline, fill gaps with synthetic low-level '
                          'Gaussian noise (matched to the recording\'s own noise floor) '
                          'instead of zeros. RECOMMENDED if the output will be re-decoded: '
                          'zero-fill causes FM-discriminator phase artifacts at burst edges '
                          'that corrupt frame sync (confirmed to drop CRC-verified blocks to '
                          'zero in testing). Noise-fill is SYNTHETIC data for cosmetic/'
                          'visualization/re-decoding purposes -- it is NOT real captured samples.')
    args = ap.parse_args()

    meta_path = Path(args.input).with_suffix('.sigmf-meta')
    iq, fs, segments, orig_start, orig_total = load_sigmf(meta_path)
    print(f'Loaded {meta_path}: {len(iq)} samples across {len(segments)} capture segment(s), '
          f'fs={fs:.0f} Hz')

    out_path = Path(args.output)

    if args.full_timeline:
        if orig_start:
            print(f'  using recorded original start time: {orig_start.isoformat()}'
                  + (f', original length {orig_total} samples' if orig_total else ''))
        else:
            print('  no original-recording metadata found; reconstructing from '
                  'first burst to last burst only (no lead-in/trailing silence)')

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
