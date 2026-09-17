# Tests

The arithmetic tests run on CPU without model weights or CUDA. Run commands
from the repository root, using Python 3.12 and public dependencies:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
mkdir -p assets/source
git clone https://github.com/guandeh17/Self-Forcing.git assets/source/Self-Forcing-main
python -m unittest discover -s tests -v
python -m compileall -q unstep runtime evaluation tests
for script in generation/*.sh evaluation/*.sh benchmarks/*.sh; do bash -n "$script"; done
```

If Self Forcing is already checked out elsewhere, set `UNSTEP_SOURCE_URI` to
that directory instead of cloning it again. The tests check that the clean-cache
noise override is applied to both input re-noising and the emitted prediction,
while preserving nonzero denoising steps and random-number consumption. The CUDA
compilation test is skipped on CPU.

For generation, GPU validation, and performance measurements, use the full
environment in [installation](../installation/). The CPU CI environment checks
correctness without measuring GPU throughput.

## Clean-cache checks

`--clean-sigma-override S` uses the same `S` in both clean-input re-noising,
`xt = (1 - S) * x + S * noise`, and the emitted prediction,
`x0 = xt - S * flow`. The DiT remains conditioned on `t=0`. Shift 5, scheduler
tables, and all nonzero denoising timesteps are unchanged. The package and both
generation launchers default to `S=0.02`.

Each process checks both coefficients before generation and records
`clean_sigma_validation`, including its `input_renoising` check. Generation
aborts if either coefficient is wrong. These checks consume no random draws
and run outside the speed timer.
