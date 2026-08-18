"""
_reconstruct.py

Shared full-timeline / compact reconstruction logic used by both
sigmf_to_iq.py and digitalrf_to_iq.py. Not a public CLI entry point --
import from one of those instead.
"""

import datetime as dt

import numpy as np


def reconstruct_compact(iq, segments, out_path):
    """Concatenate segments with no gaps. segments: list of (s0, s1, t_abs)
    into `iq`. Returns number of samples written."""
    written = 0
    with open(out_path, 'wb') as fout:
        for s0, s1, _ in segments:
            chunk = iq[s0:s1]
            chunk.tofile(fout)
            written += len(chunk)
    return written


def reconstruct_full_timeline(iq, fs, segments, out_path, orig_start=None,
                               orig_total=None, noise_fill=False):
    """Reconstruct a timeline-accurate file, inserting filler samples
    between segments sized from each segment's absolute datetime.

    segments: list of (s0, s1, t_abs) into `iq`, t_abs required (datetime).
    orig_start: datetime of the original recording's sample 0, if known
        (gives exact lead-in silence before the first burst).
    orig_total: total sample count of the original recording, if known
        (gives exact trailing silence after the last burst).
    noise_fill: fill gaps with synthetic noise matched to iq's own noise
        floor instead of zeros (recommended if the output will be
        re-decoded -- see module docstrings in sigmf_to_iq.py /
        digitalrf_to_iq.py for why plain zero-fill is not recodable-safe).

    Returns (written_samples, used_orig_bounds: bool).
    """
    if not segments or segments[0][2] is None:
        raise ValueError('reconstruct_full_timeline requires a datetime on every segment')

    noise_std = None
    if noise_fill:
        mags = np.abs(iq)
        noise_std = float(np.std(iq[mags < np.median(mags) * 1.5]))

    used_orig_bounds = orig_start is not None
    t0 = orig_start if orig_start is not None else segments[0][2]

    def make_filler(n):
        if noise_fill:
            return (np.random.normal(0, noise_std, n) +
                    1j * np.random.normal(0, noise_std, n)).astype(np.complex64)
        return np.zeros(n, dtype=np.complex64)

    written = 0
    with open(out_path, 'wb') as fout:
        cursor_sample = 0
        for s0, s1, t_abs in segments:
            target_sample = round((t_abs - t0).total_seconds() * fs)
            gap = target_sample - cursor_sample
            if gap > 0:
                filler = make_filler(gap)
                filler.tofile(fout)
                written += gap
                cursor_sample += gap
            chunk = iq[s0:s1]
            chunk.tofile(fout)
            written += len(chunk)
            cursor_sample += len(chunk)

        if orig_total is not None and cursor_sample < orig_total:
            tail = int(orig_total) - cursor_sample
            filler = make_filler(tail)
            filler.tofile(fout)
            written += tail

    return written, used_orig_bounds
