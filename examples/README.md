# Examples

`example_signal.iq` is a small (8 MB), pre-generated synthetic IQ file for
trying out the tools with zero setup. It's not modeled on any specific
real-world signal — just short tone pulses in noise, at irregular
intervals, including two bursts close together so you can see the merge/
split behavior of `--min-gap-s`.

| Parameter | Value |
|---|---|
| Sample rate | 50000 Hz |
| Duration | 20 s |
| Bursts | at 2.00s, 7.50s, 7.55s, 13.00s, 17.50s |

(the 7.50s/7.55s pair is only 0.05s apart, deliberately, to demonstrate merging)

## Try it

```bash
cd examples

# Slice it
python3 ../burst_slicer.py example_signal.iq --fs 50000 \
    --utc-start 2026-01-01T00:00:00 -o example_sliced \
    --thresh-factor 1.5 --block-s 0.01 --min-gap-s 0.5

# Expect: 4 burst regions detected (the 7.50s/7.55s pair merges into one,
# since they're closer together than --min-gap-s 0.5), ~35-40x smaller.

# Look at the timestamps
cat example_sliced.sigmf-meta

# Reconstruct it and verify you get the original back
python3 ../sigmf_to_iq.py example_sliced.sigmf-meta \
    -o example_restored.iq --full-timeline --noise-fill
```

Try lowering `--min-gap-s` below `0.05` and re-slicing — you should now get
5 separate regions instead of 4, with the close pair no longer merged.

## Regenerating / making your own

```bash
python3 generate_test_signal.py -o my_test_signal.iq --fs 100000 --duration 60
```

See `generate_test_signal.py --help` for all options. Use this if you want
a bigger file, a different sample rate, or just a fresh one with a
different random seed.
