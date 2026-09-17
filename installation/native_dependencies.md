# Native dependencies

Use the activated Python 3.12.14 environment from [installation](README.md).
Build on the deployment CPU architecture: Grace/aarch64 cannot load x86-64
native wheels. The main guide builds torchvision from its recorded source
revision; this page covers decord, which is also used by evaluation.

## decord on aarch64

PyPI's decord 0.6.0 files include Linux x86-64, Windows AMD64 and macOS x86-64
wheels, but no aarch64 wheel or source distribution. Build the native library
and Python package from the [upstream source](https://github.com/dmlc/decord#install-from-source).
This is a source-build procedure, not a claim of a tested GB200 installation.

Install a C++ compiler, `pkg-config`, CMake, and FFmpeg development libraries
(`libavcodec`, `libavformat`, `libavutil`, `libavfilter`, `libavdevice`,
`libswscale`, and `libswresample`) using your distribution's packages. The
`imageio-ffmpeg` executable does not provide these development headers/libraries.

Run from the UnStep root before installing generation requirements:

```bash
mkdir -p .build
git clone https://github.com/dmlc/decord.git .build/decord
git -C .build/decord checkout --detach v0.6.0
git -C .build/decord submodule update --init --recursive
cmake -S .build/decord -B .build/decord/build \
  -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build .build/decord/build --parallel 4
(
  cd .build/decord/python
  python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
  python -m pip install --no-deps dist/decord-*.whl
)
python -c 'import decord; print(decord.__version__, decord.__file__)'
```

`USE_CUDA=OFF` selects CPU video decoding, not CPU generation, and avoids an
unnecessary NVDEC build. The CMake policy option accommodates this older project
with the guide's CMake 4.x. If your FFmpeg development version is incompatible
with decord 0.6.0, use an upstream-supported development toolchain; do not replace
the generation torch/CUDA stack to repair the decoder build.
