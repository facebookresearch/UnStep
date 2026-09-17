# Tests

The CPU suite covers clean-cache arithmetic, generation contracts, and launcher
and evaluation safety without model weights or CUDA. Run commands from the
repository root, using Python 3.12 and public dependencies:

Use `.venv-tests`, separate from generation's `.venv`. Never install the CPU test
dependencies into a generation/CUDA environment. Set `UNSTEP_SOURCE_URI` first to
reuse a Self Forcing checkout elsewhere; otherwise the default checkout is reused
or cloned only when absent.

```bash
python3.12 -m venv .venv-tests
.venv-tests/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv-tests/bin/python -m pip install -e .
export UNSTEP_SOURCE_URI="${UNSTEP_SOURCE_URI:-$PWD/assets/source/Self-Forcing-main}"
if [ ! -d "$UNSTEP_SOURCE_URI" ]; then
  mkdir -p "$(dirname "$UNSTEP_SOURCE_URI")"
  git clone https://github.com/guandeh17/Self-Forcing.git "$UNSTEP_SOURCE_URI"
fi
.venv-tests/bin/python -m unittest discover -s tests -v
.venv-tests/bin/python -m compileall -q unstep runtime evaluation tests
for script in installation/*.sh generation/*.sh evaluation/*.sh benchmarks/*.sh; do bash -n "$script"; done
```

The tests check that the clean-cache
noise override is applied to both input re-noising and the emitted prediction,
while preserving nonzero denoising steps and random-number consumption. They also
check attention backend routing, source-reuse audits, prompt validation, launcher
cleanup and worker handling, score aggregation, and Detectron2 command selection
using stubs. No models are loaded or native extensions built. The CUDA compilation
test is skipped on CPU.

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
