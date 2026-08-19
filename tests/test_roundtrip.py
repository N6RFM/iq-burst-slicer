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

import datetime as dt
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


def expected_out_base(out_arg, utc_start_str):
    """burst_slicer.py appends '_{utc_start, normalized to hyphens}_utc' to
    whatever -o basename was given -- reproduce that here so tests know
    where to actually look for output files."""
    ts = utc_start_str
    if 'T' in ts:
        date_part, time_part = ts.split('T', 1)
        time_part = time_part.replace('-', ':')
        ts = f'{date_part}T{time_part}'
    t = dt.datetime.fromisoformat(ts)
    suffix = t.strftime('%Y-%m-%dT%H-%M-%S')
    return Path(f'{out_arg}_{suffix}_utc')


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


def run(*args, cwd=None):
    result = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=cwd)
    assert result.returncode == 0, f'command failed:\n{result.stdout}\n{result.stderr}'
    return result.stdout


UTC_START = '2026-08-18T12:00:00'


def test_slice_detects_expected_burst_count(synthetic_signal, tmp_path):
    path, fs, _ = synthetic_signal
    out_arg = tmp_path / 'sliced'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', UTC_START,
        '-o', str(out_arg),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    out_base = expected_out_base(out_arg, UTC_START)
    meta = json.loads((out_base.with_suffix('.sigmf-meta')).read_text())
    # 5 burst pulses generated, but two (8.9s, 8.95s) are only 0.05s apart,
    # well under min_gap_s=0.5, so they should merge into one region -> 4 total
    assert len(meta['captures']) == 4
    assert meta['global']['core:sample_rate'] == fs


def test_slice_produces_smaller_file(synthetic_signal, tmp_path):
    path, fs, _ = synthetic_signal
    out_arg = tmp_path / 'sliced'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', UTC_START, '-o', str(out_arg),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    out_base = expected_out_base(out_arg, UTC_START)
    original_size = path.stat().st_size
    sliced_size = out_base.with_suffix('.sigmf-data').stat().st_size
    assert sliced_size < original_size / 10  # expect at least 10x reduction for this test signal


def test_full_timeline_reconstruction_is_byte_identical(synthetic_signal, tmp_path):
    path, fs, original_sig = synthetic_signal
    out_arg = tmp_path / 'sliced'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', UTC_START, '-o', str(out_arg),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')
    out_base = expected_out_base(out_arg, UTC_START)

    restored_path = tmp_path / 'restored.iq'
    run(str(RESTORER), str(out_base.with_suffix('.sigmf-meta')),
        '-o', str(restored_path), '--full-timeline')

    restored = np.fromfile(restored_path, dtype=np.float32).view(np.complex64)

    # exact same length as the original
    assert len(restored) == len(original_sig)

    # every burst region must be byte-identical to the original
    meta = json.loads((out_base.with_suffix('.sigmf-meta')).read_text())
    t0 = None
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
    out_arg = tmp_path / 'sliced'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', UTC_START, '-o', str(out_arg),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')
    out_base = expected_out_base(out_arg, UTC_START)

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
    out_arg = tmp_path / 'sliced_drf'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', UTC_START, '-o', str(out_arg),
        '--format', 'digitalrf',
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    out_dir = expected_out_base(out_arg, UTC_START)
    info = json.loads((out_dir / 'burstslicer_info.json').read_text())
    assert len(info['captures']) == 4  # same expected merge behavior as the SigMF test
    assert info['sample_rate'] == fs


@requires_digital_rf
def test_digitalrf_full_timeline_reconstruction_is_byte_identical(synthetic_signal, tmp_path):
    path, fs, original_sig = synthetic_signal
    out_arg = tmp_path / 'sliced_drf'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', UTC_START, '-o', str(out_arg),
        '--format', 'digitalrf',
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')
    out_dir = expected_out_base(out_arg, UTC_START)

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

    sigmf_arg = tmp_path / 'sliced_sigmf'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', UTC_START, '-o', str(sigmf_arg),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')
    sigmf_base = expected_out_base(sigmf_arg, UTC_START)
    sigmf_size = sigmf_base.with_suffix('.sigmf-data').stat().st_size

    drf_arg = tmp_path / 'sliced_drf'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', UTC_START, '-o', str(drf_arg),
        '--format', 'digitalrf', '--drf-compression-level', '9',
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')
    drf_dir = expected_out_base(drf_arg, UTC_START)
    drf_size = sum(f.stat().st_size for f in drf_dir.rglob('*') if f.is_file())

    assert drf_size < sigmf_size * 1.5  # generous margin; real ratio was ~1.0


def test_utc_start_hyphen_format_matches_colon_format(synthetic_signal, tmp_path):
    """burst_slicer.py --utc-start should accept the hyphen-separated time
    format used by gr-filerepeater_n6rfm (2026-08-18T12-00-00, filename-
    safe -- no colons) and produce identical output to the equivalent
    standard ISO8601 (colon-separated) timestamp -- including an identical
    output filename suffix, since both should normalize to the same
    hyphenated form."""
    path, fs, _ = synthetic_signal

    hyphen_arg = tmp_path / 'sliced_hyphen'
    hyphen_utc = '2026-08-18T12-00-00'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', hyphen_utc, '-o', str(hyphen_arg),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    colon_arg = tmp_path / 'sliced_colon'
    colon_utc = '2026-08-18T12:00:00'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', colon_utc, '-o', str(colon_arg),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    hyphen_base = expected_out_base(hyphen_arg, hyphen_utc)
    colon_base = expected_out_base(colon_arg, colon_utc)

    # both should carry the identical timestamp suffix (just different -o prefixes)
    assert hyphen_base.name.endswith('_2026-08-18T12-00-00_utc')
    assert colon_base.name.endswith('_2026-08-18T12-00-00_utc')

    hyphen_meta = json.loads(hyphen_base.with_suffix('.sigmf-meta').read_text())
    colon_meta = json.loads(colon_base.with_suffix('.sigmf-meta').read_text())
    assert hyphen_meta == colon_meta


def test_output_filename_has_utc_suffix(synthetic_signal, tmp_path):
    """Direct check of the requested behavior: -o sliced --utc-start
    2026-08-18T12-00-00 should write sliced_2026-08-18T12-00-00_utc.*"""
    path, fs, _ = synthetic_signal
    out_arg = tmp_path / 'sliced'
    utc = '2026-08-18T12-00-00'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', utc, '-o', str(out_arg),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    expected_meta = tmp_path / 'sliced_2026-08-18T12-00-00_utc.sigmf-meta'
    expected_data = tmp_path / 'sliced_2026-08-18T12-00-00_utc.sigmf-data'
    assert expected_meta.exists()
    assert expected_data.exists()


def test_utc_offset_matches_equivalent_utc_start(synthetic_signal, tmp_path):
    """--utc-offset, reading a local timestamp embedded in a
    gr-filerepeater_n6rfm-style filename, should produce identical output
    to specifying the equivalent UTC time directly with --utc-start."""
    orig_path, fs, _ = synthetic_signal

    # rename to embed a local timestamp the way gr-filerepeater_n6rfm does
    filename_path = tmp_path / 'DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq'
    filename_path.write_bytes(orig_path.read_bytes())

    offset_arg = tmp_path / 'sliced_offset'
    run(str(SLICER), str(filename_path), '--fs', str(fs),
        '--utc-offset', '-4', '-o', str(offset_arg),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    # 13:30 local, UTC-4 -> 17:30 UTC
    equiv_utc = '2026-08-18T17-30-00'
    start_arg = tmp_path / 'sliced_start'
    run(str(SLICER), str(orig_path), '--fs', str(fs),
        '--utc-start', equiv_utc, '-o', str(start_arg),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    offset_base = expected_out_base(offset_arg, equiv_utc)
    start_base = expected_out_base(start_arg, equiv_utc)

    offset_meta = json.loads(offset_base.with_suffix('.sigmf-meta').read_text())
    start_meta = json.loads(start_base.with_suffix('.sigmf-meta').read_text())
    assert offset_meta == start_meta


def test_utc_offset_without_embedded_timestamp_fails_clearly(synthetic_signal, tmp_path):
    """--utc-offset on a filename with no embedded timestamp should fail
    with a clear, actionable error rather than crash or silently guess."""
    path, fs, _ = synthetic_signal
    result = subprocess.run(
        [sys.executable, str(SLICER), str(path), '--fs', str(fs),
         '--utc-offset', '-4', '-o', str(tmp_path / 'sliced')],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'no embedded timestamp' in result.stderr
    assert '--utc-start' in result.stderr


def test_no_output_arg_reuses_input_filename_replacing_embedded_timestamp(synthetic_signal, tmp_path):
    """With no -o given and an input filename that already embeds a
    gr-filerepeater_n6rfm-style local timestamp, the output filename should
    reuse the input name with that timestamp REPLACED by the corrected UTC
    time (not a second, redundant timestamp appended)."""
    orig_path, fs, _ = synthetic_signal
    filename_path = tmp_path / 'DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq'
    filename_path.write_bytes(orig_path.read_bytes())

    run(str(SLICER), str(filename_path), '--fs', str(fs),
        '--utc-offset', '-4',
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5',
        cwd=tmp_path)

    expected = tmp_path / 'DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T17-30-00_utc.sigmf-meta'
    assert expected.exists()
    # the OLD (local) timestamp must not appear in any OUTPUT filename
    # (the input .iq file itself legitimately still has it -- that's fine)
    outputs_with_old_ts = [p for p in tmp_path.glob('*T13-30-00*') if p.suffix != '.iq']
    assert not outputs_with_old_ts


def test_no_output_arg_falls_back_to_appended_suffix(synthetic_signal, tmp_path):
    """With no -o given and an input filename with NO embedded timestamp,
    fall back to appending the usual _{utc}_utc suffix to the input stem."""
    path, fs, _ = synthetic_signal  # fixture writes to 'synthetic.iq', no embedded timestamp

    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', '2026-08-18T17-48-10',
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5',
        cwd=tmp_path)

    expected = tmp_path / 'synthetic_2026-08-18T17-48-10_utc.sigmf-meta'
    assert expected.exists()


def test_format_defaults_to_sigmf_without_flag(synthetic_signal, tmp_path):
    """--format should not be required; omitting it should behave exactly
    like --format sigmf."""
    path, fs, _ = synthetic_signal
    out_arg = tmp_path / 'sliced'
    run(str(SLICER), str(path), '--fs', str(fs),
        '--utc-start', UTC_START, '-o', str(out_arg),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')
    out_base = expected_out_base(out_arg, UTC_START)
    assert out_base.with_suffix('.sigmf-data').exists()
    assert out_base.with_suffix('.sigmf-meta').exists()


def test_no_time_arg_reuses_embedded_timestamp_without_utc_claim(synthetic_signal, tmp_path):
    """With neither --utc-start nor --utc-offset given, and an input
    filename embedding a gr-filerepeater_n6rfm-style timestamp, that
    timestamp should be reused EXACTLY as-is, with NO timezone/UTC
    assumption -- the output filename should be identical to the input
    filename's stem (no '_utc' marker, no substitution), since nothing
    was actually verified or converted."""
    orig_path, fs, _ = synthetic_signal
    filename_path = tmp_path / 'DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T17-30-00.iq'
    filename_path.write_bytes(orig_path.read_bytes())

    run(str(SLICER), str(filename_path), '--fs', str(fs),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5',
        cwd=tmp_path)

    # output name == input stem, completely unchanged, no '_utc' appended
    expected = tmp_path / 'DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T17-30-00.sigmf-meta'
    assert expected.exists()
    assert not list(tmp_path.glob('*_utc*'))


def test_no_time_arg_without_embedded_timestamp_fails_clearly(synthetic_signal, tmp_path):
    """With neither --utc-start, --utc-offset, nor a filename-embedded
    timestamp, fail with a clear, actionable error rather than crash."""
    path, fs, _ = synthetic_signal
    result = subprocess.run(
        [sys.executable, str(SLICER), str(path), '--fs', str(fs)],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'no embedded timestamp' in result.stderr
    assert '--utc-start' in result.stderr and '--utc-offset' in result.stderr


def test_no_time_arg_with_explicit_output_uses_it_unmodified(synthetic_signal, tmp_path):
    """With neither --utc-start nor --utc-offset given, but -o given
    explicitly, the given name should be used exactly as-is, with no
    '_utc'-claiming timestamp appended (nothing was verified as UTC)."""
    orig_path, fs, _ = synthetic_signal
    filename_path = tmp_path / 'DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq'
    filename_path.write_bytes(orig_path.read_bytes())

    run(str(SLICER), str(filename_path), '--fs', str(fs), '-o', str(tmp_path / 'manual_name'),
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5')

    assert (tmp_path / 'manual_name.sigmf-meta').exists()
    assert (tmp_path / 'manual_name.sigmf-data').exists()
    assert not list(tmp_path.glob('manual_name*_utc*'))


def test_utc_start_and_utc_offset_still_mutually_exclusive(synthetic_signal, tmp_path):
    """Both being optional now shouldn't loosen the still-required
    mutual exclusivity between the two when both ARE given."""
    path, fs, _ = synthetic_signal
    result = subprocess.run(
        [sys.executable, str(SLICER), str(path), '--fs', str(fs),
         '--utc-start', '2026-08-18T12-00-00', '--utc-offset', '-4',
         '-o', str(tmp_path / 'sliced')],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'not allowed with argument' in result.stderr


def test_utc_offset_still_appends_utc_suffix(synthetic_signal, tmp_path):
    """Sanity check that fixing the 'neither given' case didn't disturb
    the (still correct, still verified) --utc-offset behavior."""
    orig_path, fs, _ = synthetic_signal
    filename_path = tmp_path / 'DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00.iq'
    filename_path.write_bytes(orig_path.read_bytes())

    run(str(SLICER), str(filename_path), '--fs', str(fs), '--utc-offset', '-4',
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5',
        cwd=tmp_path)

    expected = tmp_path / 'DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T17-30-00_utc.sigmf-meta'
    assert expected.exists()


def test_default_output_no_doubled_utc_suffix(synthetic_signal, tmp_path):
    """Feeding back a filename that already ends in '_utc' (e.g. this
    tool's own previous output, or a manually-named file), WITH
    --utc-offset given, should not produce a doubled '_utc_utc' in the
    output name."""
    orig_path, fs, _ = synthetic_signal
    filename_path = tmp_path / 'DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T13-30-00_utc.iq'
    filename_path.write_bytes(orig_path.read_bytes())

    run(str(SLICER), str(filename_path), '--fs', str(fs), '--utc-offset', '-4',
        '--thresh-factor', '1.5', '--block-s', '0.01', '--min-gap-s', '0.5',
        cwd=tmp_path)

    expected = tmp_path / 'DSTARONESPARROW_50000SPS_435700000Hz_2026_08_18_T17-30-00_utc.sigmf-meta'
    assert expected.exists()
    assert not list(tmp_path.glob('*_utc_utc*'))
