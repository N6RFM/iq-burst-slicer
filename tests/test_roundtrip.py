"""
Round-trip tests for burst_slicer.py + sigmf_to_iq.py + digitalrf_to_iq.py.

Uses a synthetic signal deliberately unlike any specific real-world protocol
this tool was developed against (different sample rate, different
modulation, much shorter bursts, irregular spacing including two bursts
close enough together to exercise the merge-vs-split logic) to verify the
tools are genuinely general-purpose burst detectors/reconstructors, not
tuned to one signal.

Digital RF tests are skipped automatically if the `digital_rf` package
isn't installed (it's an optional dependency -- only needed for
--format digitalrf).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SLICER = REPO_ROOT / 'burst_slicer.py'
RESTORER = REPO_ROOT / 'sigmf_to_iq.py'
DRF_RESTORER = REPO_ROOT / 'digitalrf_to_iq.py'

try:
    import digital_rf  # noqa: F401
    HAVE_DIGITAL_RF = True
except ImportError:
    HAVE_DIGITAL_RF = False

requires_digital_rf = pytest.mark.skipif(
    not HAVE_DIGITAL_RF, reason='digital_rf package not installed (optional dependency)')


@pytest.fixture
def synthetic_signal(tmp_path):
    """Generates a synthetic bursty IQ file: irregular OOK tone pulses in
    noise, including two bursts close together to test merge behavior."""
    rng = np.random.default_rng(42)
    fs = 200000
    duration_s = 30
    n = int(duration_s * fs)

    noise = (rng.normal(0, 0.01, n) + 1j * rng.normal(0, 0.01, n)).astype(np.complex64)

    burst_times_s = [2.3, 8.9, 8.95, 19.4, 27.1]  # last two of the first group are close together
    burst_dur_s = 0.08
    tone_freq = 15000

    sig = noise.copy()
    for t0 in burst_times_s:
        s0 = int(t0 * fs)
        s1 = int((t0 + burst_dur_s) * fs)
        tt = np.arange(s1 - s0) / fs
        tone = 0.3 * np.exp(1j * 2 * np.pi * tone_freq * tt).astype(np.complex64)
        sig[s0:s1] += tone

    path = tmp_path / 'synthetic.iq'
    sig.tofile(path)
    return path, fs, sig


def run(*args):
    result = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    assert result.returncode == 0, f'command failed:\n{result.stdout}\n{result.stderr}'
    return result.stdout


def test_slice_detects_expected_burst_count(synthetic_signal, tmp_path):
    path, fs, _ = synthetic_signal
    out_base = tmp_path / 'sliced'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', '2026-08-18T12:00:00',
        '-o', str(out_base),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    meta = json.loads((out_base.with_suffix('.sigmf-meta')).read_text())
    # 5 burst pulses generated, but two (8.9s, 8.95s) are only 0.05s apart,
    # well under min_gap_s=0.5, so they should merge into one region -> 4 total
    assert len(meta['captures']) == 4
    assert meta['global']['core:sample_rate'] == fs


def test_slice_produces_smaller_file(synthetic_signal, tmp_path):
    path, fs, _ = synthetic_signal
    out_base = tmp_path / 'sliced'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', '2026-08-18T12:00:00', '-o', str(out_base),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    original_size = path.stat().st_size
    sliced_size = out_base.with_suffix('.sigmf-data').stat().st_size
    assert sliced_size < original_size / 10  # expect at least 10x reduction for this test signal


def test_full_timeline_reconstruction_is_byte_identical(synthetic_signal, tmp_path):
    path, fs, original_sig = synthetic_signal
    out_base = tmp_path / 'sliced'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', '2026-08-18T12:00:00', '-o', str(out_base),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    restored_path = tmp_path / 'restored.iq'
    run(str(RESTORER), str(out_base.with_suffix('.sigmf-meta')),
        '-o', str(restored_path), '--full-timeline')

    restored = np.fromfile(restored_path, dtype=np.float32).view(np.complex64)

    # exact same length as the original
    assert len(restored) == len(original_sig)

    # every burst region must be byte-identical to the original
    meta = json.loads((out_base.with_suffix('.sigmf-meta')).read_text())
    t0 = None
    import datetime as dt
    for cap in meta['captures']:
        t_abs = dt.datetime.fromisoformat(cap['core:datetime'].replace('Z', '+00:00'))
        if t0 is None:
            t0 = dt.datetime.fromisoformat(
                meta['global']['burstslicer:original_utc_start'].replace('Z', '+00:00'))
        offset_samples = round((t_abs - t0).total_seconds() * fs)
        n_samples = cap['core:sample_count']
        np.testing.assert_array_equal(
            restored[offset_samples:offset_samples + n_samples],
            original_sig[offset_samples:offset_samples + n_samples],
        )


def test_compact_reconstruction_contains_only_burst_samples(synthetic_signal, tmp_path):
    path, fs, _ = synthetic_signal
    out_base = tmp_path / 'sliced'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', '2026-08-18T12:00:00', '-o', str(out_base),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    compact_path = tmp_path / 'compact.iq'
    run(str(RESTORER), str(out_base.with_suffix('.sigmf-meta')),
        '-o', str(compact_path), '--compact')

    sliced_data = np.fromfile(out_base.with_suffix('.sigmf-data'),
                               dtype=np.float32).view(np.complex64)
    compact_data = np.fromfile(compact_path, dtype=np.float32).view(np.complex64)
    np.testing.assert_array_equal(sliced_data, compact_data)


@requires_digital_rf
def test_digitalrf_slice_detects_expected_burst_count(synthetic_signal, tmp_path):
    path, fs, _ = synthetic_signal
    out_dir = tmp_path / 'sliced_drf'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', '2026-08-18T12:00:00', '-o', str(out_dir),
        '--format', 'digitalrf',
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    info = json.loads((out_dir / 'burstslicer_info.json').read_text())
    assert len(info['captures']) == 4  # same expected merge behavior as the SigMF test
    assert info['sample_rate'] == fs


@requires_digital_rf
def test_digitalrf_full_timeline_reconstruction_is_byte_identical(synthetic_signal, tmp_path):
    path, fs, original_sig = synthetic_signal
    out_dir = tmp_path / 'sliced_drf'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', '2026-08-18T12:00:00', '-o', str(out_dir),
        '--format', 'digitalrf',
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    restored_path = tmp_path / 'restored_drf.iq'
    run(str(DRF_RESTORER), str(out_dir), '-o', str(restored_path), '--full-timeline')

    restored = np.fromfile(restored_path, dtype=np.float32).view(np.complex64)
    assert len(restored) == len(original_sig)

    info = json.loads((out_dir / 'burstslicer_info.json').read_text())
    for cap in info['captures']:
        s0 = cap['sample_start']
        n = cap['sample_count']
        np.testing.assert_array_equal(
            restored[s0:s0 + n], original_sig[s0:s0 + n])


@requires_digital_rf
def test_digitalrf_and_sigmf_produce_comparable_size(synthetic_signal, tmp_path):
    """Regression guard for the corrected comparison: with compression
    enabled, Digital RF should NOT be dramatically larger than SigMF for
    this kind of sparse-burst signal (they were found to be within ~1% of
    each other in real testing -- see README). This test just guards
    against a large regression, not an exact ratio."""
    path, fs, _ = synthetic_signal

    sigmf_base = tmp_path / 'sliced_sigmf'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', '2026-08-18T12:00:00', '-o', str(sigmf_base),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')
    sigmf_size = sigmf_base.with_suffix('.sigmf-data').stat().st_size

    drf_dir = tmp_path / 'sliced_drf'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', '2026-08-18T12:00:00', '-o', str(drf_dir),
        '--format', 'digitalrf', '--drf-compression-level', '9',
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')
    drf_size = sum(f.stat().st_size for f in drf_dir.rglob('*') if f.is_file())

    assert drf_size < sigmf_size * 1.5  # generous margin; real ratio was ~1.0
