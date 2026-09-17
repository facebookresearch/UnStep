# Contributing to UnStep

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

Submit changes through GitHub pull requests. Maintainers review the changes
and test results before merging. Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite and syntax checks below pass.
5. Follow the existing code style.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Tests

The arithmetic tests run on CPU without model weights or CUDA. Use Python 3.12
and public dependencies:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
mkdir -p assets/source
git clone https://github.com/guandeh17/Self-Forcing.git assets/source/Self-Forcing-main
python -m unittest discover -s tests -v
python -m compileall -q unstep runtime scripts tests
for script in scripts/*.sh; do bash -n "$script"; done
```

If Self Forcing is already checked out elsewhere, set `UNSTEP_SOURCE_URI` to
that directory instead of cloning it again. The tests check that the clean-cache
noise override is applied to both input re-noising and the emitted prediction,
while preserving nonzero denoising steps and random-number consumption. The CUDA
compilation test is skipped on CPU.

For generation, GPU validation, and performance measurements, use the full
environment in [SETUP_H100.md](SETUP_H100.md). The CPU CI environment does not
reproduce H100 throughput.

## Coding Style

* Use four spaces for Python indentation.
* Follow the surrounding code's formatting and naming.
* Include the copyright and license header used by existing source files.

## License

By contributing to UnStep, you agree that your contributions will be licensed
under the [LICENSE](LICENSE) file in the root directory of this source tree.
