#!/usr/bin/env python3
"""
generate_test_signal.py

Generates a synthetic bursty IQ file for trying out burst_slicer.py and
sigmf_to_iq.py with zero setup. Not modeled on any specific real-world
protocol -- just short OOK (on-off keyed) tone pulses in noise, at
irregular intervals, including two bursts close together so you can see
the --min-gap-s merge-vs-split behavior in action.

The repo ships a small pre-generated file at examples/example_signal.iq
(see examples/README.md for how to use it); run this script yourself if
you want a different size/duration, or to regenerate it.

Usage:
    python3 generate_test_signal.py -o example_signal.iq
    python3 generate_test_signal.py -o bigger_example.iq --duration 60 --fs 100000

Requires: numpy
"""

import argparse
from pathlib import Path

import numpy as np


def generate(fs=50000, duration_s=20, seed=42):
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    noise = (rng.normal(0, 0.01, n) + 1j * rng.normal(0, 0.01, n)).astype(np.complex64)

    # irregular burst times, including two close together near the middle
    burst_times_s = [t for t in [2.0, 7.5, 7.55, 13.0, duration_s - 2.5] if t < duration_s - 0.2]
    burst_dur_s = 0.08
    tone_freq = fs * 0.075  # comfortably inside the band regardless of fs

    sig = noise.copy()
    for t0 in burst_times_s:
        s0 = int(t0 * fs)
        s1 = int((t0 + burst_dur_s) * fs)
        tt = np.arange(s1 - s0) / fs
        tone = 0.3 * np.exp(1j * 2 * np.pi * tone_freq * tt).astype(np.complex64)
        sig[s0:s1] += tone

    return sig, burst_times_s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--output', default='example_signal.iq', help='Output IQ file path')
    ap.add_argument('--fs', type=float, default=50000, help='Sample rate (default 50000)')
    ap.add_argument('--duration', type=float, default=20, help='Total duration in seconds (default 20)')
    ap.add_argument('--seed', type=int, default=42, help='Random seed (default 42, for reproducibility)')
    args = ap.parse_args()

    sig, burst_times = generate(fs=args.fs, duration_s=args.duration, seed=args.seed)
    out_path = Path(args.output)
    sig.tofile(out_path)

    print(f'Wrote {out_path}  ({len(sig)} samples, {len(sig)/args.fs:.2f}s, '
          f'{out_path.stat().st_size} bytes, fs={args.fs:.0f} Hz)')
    print(f'Bursts at: {", ".join(f"{t:.2f}s" for t in burst_times)}')
    print('(the pair close together tests --min-gap-s merge-vs-split behavior)')


if __name__ == '__main__':
    main()
