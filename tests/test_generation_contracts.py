# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only generation, staging, and attention dispatch contracts."""

import builtins
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel
import zipfile

import torch

from unstep import wrapper_runtime as runtime
from unstep.wrapper import OfflineSelfForcingWrapper, WrapperConfig


# Driver contracts do not require torchvision's optional video dependencies.
_torchvision = ModuleType("torchvision")
_torchvision.io = ModuleType("torchvision.io")
_torchvision.io.write_video = Mock()
with patch.dict(sys.modules, {"torchvision": _torchvision, "torchvision.io": _torchvision.io}):
    runner = importlib.import_module("unstep.run_wrapper_generation")

SF_SOURCE = Path(os.environ.get("UNSTEP_SOURCE_URI", "assets/source/Self-Forcing-main"))


class CpuTestCase(unittest.TestCase):
    def setUp(self):
        guard = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden"))
        guard.start()
        self.addCleanup(guard.stop)


class DriverContracts(CpuTestCase):
    def test_prompt_bounds_and_uniqueness(self):
        self.assertEqual(runner.parse_indices("all", 3), [0, 1, 2])
        self.assertEqual(runner.parse_indices("2, 0", 3), [2, 0])
        for value in ("-1", "3", "0,0", ""):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                runner.parse_indices(value, 3)

    def test_config_preserves_hopper_defaults_and_serializes_seed_mode(self):
        args = SimpleNamespace(seed_mode="sequential_global", disable_block_schedule_overrides=False,
                               clean_sigma_override=0.02)
        self.assertEqual(runner.make_wrapper_config(args, "h100"), WrapperConfig())
        args.seed_mode = "per_prompt"
        cfg = runner.make_wrapper_config(args, "fa2")
        self.assertEqual(json.loads(json.dumps(cfg.base.__dict__))["seed_mode"], "per_prompt")
        self.assertFalse(cfg.use_fa3_batch1)
        self.assertFalse(cfg.use_fa3_slice_lens)
        args.disable_block_schedule_overrides = True
        args.clean_sigma_override = 0.0
        cfg = runner.make_wrapper_config(args, "fa2")
        self.assertEqual(cfg.block_schedule_overrides, ())
        self.assertEqual(cfg.clean_sigma_override, 0.0)

    def test_cli_accepts_fa2_without_loading_a_model(self):
        was_enabled = torch.is_grad_enabled()
        self.addCleanup(torch.set_grad_enabled, was_enabled)
        with patch.object(sys, "argv", ["unstep", "--fa3-import-backend", "fa2"]), \
                patch.object(runtime, "resolve_attention_backend", side_effect=RuntimeError("parsed")) as select:
            with self.assertRaisesRegex(RuntimeError, "parsed"):
                runner.main()
        select.assert_called_once_with("fa2")

    def test_prepare_args_receive_effective_backend(self):
        args = SimpleNamespace(source_uri="source", weights_uri="weights", vae_uri="vae",
                               checkpoint_uri=None)
        for backend in ("fa2", "h100", "flash_attn_3"):
            prepared = runner.make_prepare_args(args, WrapperConfig(), backend)
            self.assertEqual(prepared.fa3_import_backend, backend)
            self.assertEqual(prepared.enable_flash_attn3, backend != "fa2")
            self.assertEqual(prepared.timestep_shift, 5.0)

    def test_default_schedule_is_unchanged(self):
        wrapper = OfflineSelfForcingWrapper()
        pipeline = SimpleNamespace(denoising_step_list=[1000, 900, 800, 600])
        schedules = [wrapper.resolve_timesteps(pipeline, block) for block in range(7)]
        self.assertEqual(schedules, [[1000, 900, 800, 600, 0], [1000, 600, 0],
                                    [1000, 600, 0], [1000, 800, 0], [1000, 800, 0],
                                    [1000, 800, 0], [1000, 600, 0]])
        self.assertEqual(sum(map(len, schedules)), 23)

    def test_cleanup_rejects_protected_ancestors_and_descendants(self):
        protected = runner.ROOT / "results"
        with patch.object(runner, "_ORIGINAL_RMTREE") as remove:
            for path in (protected, protected / "run", runner.ROOT, runner.ROOT.parent, Path("/")):
                with self.subTest(path=path), self.assertRaises(RuntimeError):
                    runner.guarded_rmtree(path)
            remove.assert_not_called()
            runner.guarded_rmtree(Path("/tmp/unstep-test-cleanup"))
            remove.assert_called_once()

    def test_speed_cleanup_preserves_json_and_repository(self):
        video = Path("/tmp/unstep-test-run/videos")
        with self.assertRaisesRegex(RuntimeError, "out-json"):
            runner.assert_safe_speed_video_cleanup_dir(
                video, run_name="unstep-test-run", out_json=video / "scores.json")
        runner.assert_safe_speed_video_cleanup_dir(
            video, run_name="unstep-test-run", out_json=video.parent / "scores.json")
        with self.assertRaisesRegex(RuntimeError, "protected"):
            runner.assert_safe_speed_video_cleanup_dir(
                runner.ROOT, run_name=runner.ROOT.name, out_json=Path("/tmp/scores.json"))
        for path in (runner.ROOT / "unstep", runner.ROOT / "tests"):
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "protected"):
                runner.assert_safe_speed_video_cleanup_dir(
                    path, run_name=path.name, out_json=Path("/tmp/scores.json"))
        with self.assertRaisesRegex(RuntimeError, "protected"):
            runner.assert_safe_speed_video_cleanup_dir(
                video, run_name="unstep-test-run", out_json=video.parent / "scores.json",
                input_paths=(video / "weights.pt",))
        runner.assert_safe_speed_video_cleanup_dir(
            runner.ROOT / "results/unstep-test-run/videos", run_name="unstep-test-run",
            out_json=runner.ROOT / "results/unstep-test-run/scores.json")

    def test_speed_cleanup_rejects_results_alias_into_code(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            (repository / "unstep").mkdir(parents=True)
            (repository / "results").symlink_to(repository / "unstep", target_is_directory=True)
            with patch.object(runner, "ROOT", repository), self.assertRaisesRegex(RuntimeError, "protected"):
                runner.assert_safe_speed_video_cleanup_dir(
                    repository / "results/run/videos", run_name="run", out_json=Path(directory) / "scores.json")

    def test_ffmpeg_fallback_uses_exported_or_imageio_executable(self):
        frames_np = SimpleNamespace(shape=(2, 2, 2, 3), tobytes=lambda: b"pixels")
        frames = Mock(dtype=torch.uint8, device=torch.device("cpu"))
        frames.contiguous.return_value.numpy.return_value = frames_np
        video = Mock()
        video.shape = (1, 2, 2, 2, 3)
        video.__getitem__ = Mock(return_value=frames)
        imageio = ModuleType("imageio_ffmpeg")
        imageio.get_ffmpeg_exe = Mock(return_value="/imageio/ffmpeg")
        with tempfile.TemporaryDirectory() as directory:
            for configured, expected in (("/exported/ffmpeg", "/exported/ffmpeg"), ("", "/imageio/ffmpeg")):
                with self.subTest(configured=configured), \
                        patch.dict(os.environ, {"IMAGEIO_FFMPEG_EXE": configured}), \
                        patch.dict(sys.modules, {"imageio_ffmpeg": imageio}), \
                        patch.object(runner, "write_video", side_effect=RuntimeError("write_video is unavailable")), \
                        patch.object(runner.subprocess, "run") as process:
                    runner.save_video(Path(directory) / "output.mp4", video, 16)
                    self.assertEqual(process.call_args.args[0][0], expected)
                    self.assertEqual(process.call_args.kwargs, {"input": b"pixels", "check": True})
        imageio.get_ffmpeg_exe.assert_called_once()


class AttentionContracts(CpuTestCase):
    def setUp(self):
        super().setUp()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.attention_path = self.root / "wan/modules/attention.py"
        self.attention_path.parent.mkdir(parents=True)
        source = SF_SOURCE / "wan/modules/attention.py"
        if not source.is_file():
            self.skipTest("set UNSTEP_SOURCE_URI to upstream Self Forcing")
        self.attention_path.write_text(source.read_text())

    def test_backend_selection(self):
        for capability, expected in (((9, 0), "h100"), ((10, 0), "fa2"), ((8, 0), "fa2")):
            self.assertEqual(runtime.resolve_attention_backend("auto", capability=capability), expected)
        self.assertEqual(runtime.resolve_attention_backend("fa2", capability=(9, 0)), "fa2")
        for requested in ("h100", "flash_attn_3"):
            with self.assertRaisesRegex(RuntimeError, "Hopper"):
                runtime.resolve_attention_backend(requested, capability=(10, 0))

    def load_attention(self, backend, capability, *, fa3_available=True):
        fake_fa2 = ModuleType("flash_attn")
        fake_fa2.flash_attn_varlen_func = Mock(return_value=torch.zeros(6, 2, 4))
        fake_fa3 = ModuleType("flash_attn_interface")
        fake_fa3.flash_attn_func = Mock(return_value=torch.zeros(1, 6, 2, 4))
        fake_fa3.flash_attn_varlen_func = Mock(return_value=torch.zeros(6, 2, 4))
        fake_fa3_package = ModuleType("flash_attn_3")
        fake_fa3_package.flash_attn_interface = fake_fa3
        module = ModuleType("wan.modules.attention")
        original_import = builtins.__import__
        imports = []

        def tracked_import(name, *args, **kwargs):
            imports.append(name)
            if capability[0] != 9 and name.startswith(("flash_attn_interface", "flash_attn_3")):
                raise AssertionError("FA3 imported on FA2 path")
            return original_import(name, *args, **kwargs)

        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "get_device_capability", return_value=capability), \
                patch.dict(sys.modules, {
                    "flash_attn": fake_fa2,
                    "flash_attn_interface": fake_fa3 if fa3_available else None,
                    "flash_attn_3": fake_fa3_package if fa3_available else None,
                    "flash_attn_3.flash_attn_interface": fake_fa3 if fa3_available else None,
                }), \
                patch.object(builtins, "__import__", side_effect=tracked_import):
            runtime.patch_attention_fallback(self.root, enable_flash_attn3=backend != "fa2",
                                             fa3_import_backend=backend)
            exec(compile(self.attention_path.read_text(), str(self.attention_path), "exec"), module.__dict__)
        return module, imports

    def test_fa2_never_imports_fa3_and_preserves_public_fa_version(self):
        module, imports = self.load_attention("fa2", (10, 0))
        self.assertFalse(module.FLASH_ATTN_3_AVAILABLE)
        self.assertTrue(module.FLASH_ATTN_2_AVAILABLE)
        self.assertNotIn("flash_attn_interface", imports)
        with patch.object(module, "flash_attention", return_value=sentinel.output) as call:
            result = module.attention(None, None, None, fa_version=2)
        self.assertIs(result, sentinel.output)
        self.assertEqual(call.call_args.kwargs["version"], 2)

    def test_hopper_imports_fa3(self):
        module, imports = self.load_attention("h100", (9, 0))
        self.assertTrue(module.FLASH_ATTN_3_AVAILABLE)
        self.assertIn("flash_attn_interface", imports)

    def test_hopper_selection_does_not_fall_back_when_fa3_is_missing(self):
        native = self.attention_path.read_text()
        for backend in ("auto", "h100", "flash_attn_3"):
            with self.subTest(backend=backend):
                self.attention_path.write_text(native)
                with self.assertRaises(ModuleNotFoundError):
                    self.load_attention(backend, (9, 0), fa3_available=False)

    def test_auto_hopper_backend_agrees_with_runtime_info(self):
        attention, _ = self.load_attention("auto", (9, 0))
        self.assertEqual(attention._unstep_attention_backend, "h100")
        wan, modules = ModuleType("wan"), ModuleType("wan.modules")
        wan.modules, modules.attention = modules, attention
        with patch.dict(sys.modules, {"wan": wan, "wan.modules": modules,
                                      "wan.modules.attention": attention}), \
                patch.object(runtime.importlib.util, "find_spec", return_value=sentinel.spec), \
                patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), \
                patch.object(torch.cuda, "get_device_name", return_value="H100 test device"):
            info = runtime.runtime_info(self.root, "bf16", attention_backend="h100")
        self.assertTrue(info["wan_flash_attn_3_available"])
        self.assertEqual(info["gpu_capability"], (9, 0))
        self.assertNotIn("wan_attention_import_error", info)

    def test_original_h100_import_header_is_supported(self):
        native = self.attention_path.read_text().replace("is_hopper_gpu", "is_h100_gpu")
        self.attention_path.write_text(native)
        attention, _ = self.load_attention("h100", (9, 0))
        self.assertTrue(attention.FLASH_ATTN_3_AVAILABLE)

    def test_explicit_fa2_on_hopper_does_not_import_fa3(self):
        module, imports = self.load_attention("fa2", (9, 0))
        self.assertEqual(module._unstep_attention_backend, "fa2")
        self.assertFalse(module.FLASH_ATTN_3_AVAILABLE)
        self.assertNotIn("flash_attn_interface", imports)

    def test_auto_gb200_suppresses_fa3_before_import(self):
        module, imports = self.load_attention("auto", (10, 0))
        self.assertEqual(module._unstep_attention_backend, "fa2")
        self.assertFalse(module.FLASH_ATTN_3_AVAILABLE)
        self.assertNotIn("flash_attn_interface", imports)

    def test_diagnostic_fa2_path_does_not_probe_fa3_imports(self):
        with patch.object(runtime.importlib.util, "find_spec", return_value=None) as find, \
                patch.object(torch.cuda, "is_available", return_value=False):
            info = runtime.runtime_info(self.root, "bf16", attention_backend="fa2")
        self.assertIsNone(info["flash_attn_3_spec"])
        self.assertEqual([call.args[0] for call in find.call_args_list], ["flash_attn"])

    def test_cached_dispatch_and_varlen_tensor_tuple_returns(self):
        class Input:
            device = torch.device("cuda:0")
            dtype = torch.bfloat16

            def __init__(self, shape=(1, 6, 2, 4)):
                self.shape = shape

            def size(self, axis):
                return self.shape[axis]

            def to(self, dtype):
                return self

            def flatten(self, start, end):
                return Input((self.shape[0] * self.shape[1], *self.shape[2:]))

            def __getitem__(self, indices):
                shape = list(self.shape)
                for axis, index in enumerate(indices):
                    shape[axis] = len(range(shape[axis])[index])
                return Input(tuple(shape))

        full, arange = torch.full, torch.arange
        cases = [(True, True, None, "dense", False),
                 (True, True, 2, "fa2", False),
                 (False, True, None, "fa3", False),
                 (False, True, None, "fa3", True),
                 (True, False, None, "fa2", False)]
        for dense, available, version, expected, tuple_output in cases:
            with self.subTest(dense=dense, available=available, version=version, tuple=tuple_output):
                attention = ModuleType("wan.modules.attention")
                attention.FLASH_ATTN_3_AVAILABLE = available
                attention.flash_attention = Mock(return_value=sentinel.fallback)
                output = torch.zeros(6, 2, 4)
                attention.flash_attn_interface = SimpleNamespace(
                    flash_attn_func=Mock(return_value=torch.zeros(1, 6, 2, 4)),
                    flash_attn_varlen_func=Mock(return_value=(output,) if tuple_output else output))
                attention.flash_attn = SimpleNamespace(flash_attn_varlen_func=Mock(return_value=output))
                wan, modules, model = ModuleType("wan"), ModuleType("wan.modules"), ModuleType("wan.modules.model")
                wan.modules, modules.attention, modules.model = modules, attention, model
                with patch.dict(sys.modules, {"wan": wan, "wan.modules": modules,
                                              "wan.modules.attention": attention, "wan.modules.model": model}), \
                        patch.object(torch, "full", side_effect=lambda *a, **kw: full(*a, **{**kw, "device": "cpu"})), \
                        patch.object(torch, "arange", side_effect=lambda *a, **kw: arange(*a, **{**kw, "device": "cpu"})):
                    runtime.install_flash_attention_seqlen_cache(
                        use_fixed_fa3_batch1=dense, use_fixed_fa3_batch1_slice_lens=True)
                    actual = attention.flash_attention(Input(), Input(), Input(), version=version)
                    self.assertEqual(tuple(actual.shape), (1, 6, 2, 4))
                    calls = {"dense": attention.flash_attn_interface.flash_attn_func.call_count,
                             "fa3": attention.flash_attn_interface.flash_attn_varlen_func.call_count,
                             "fa2": attention.flash_attn.flash_attn_varlen_func.call_count}
                    self.assertEqual(calls, {name: int(name == expected) for name in calls})
                    attention.flash_attention(Input(), Input(), Input(), k_lens=[4], version=2)
                    self.assertEqual(attention._sf_original_flash_attention.call_args.kwargs["version"], 2)
                    if expected == "dense":
                        attention.flash_attention(Input(), Input(), Input(), k_lens=[4])
                        self.assertEqual(attention.flash_attn_interface.flash_attn_func.call_count, 2)
                    installed = attention.flash_attention
                    runtime.install_flash_attention_seqlen_cache(
                        use_fixed_fa3_batch1=False, use_fixed_fa3_batch1_slice_lens=False)
                    self.assertIs(attention.flash_attention, installed)
                    for call in (attention.flash_attn_interface.flash_attn_func,
                                 attention.flash_attn_interface.flash_attn_varlen_func,
                                 attention.flash_attn.flash_attn_varlen_func):
                        call.reset_mock()
                    for _ in range(2):
                        actual = attention.flash_attention(Input(), Input(), Input(), version=2)
                        self.assertEqual(tuple(actual.shape), (1, 6, 2, 4))
                        self.assertEqual(actual.dtype, torch.bfloat16)
                    attention.flash_attn_interface.flash_attn_func.assert_not_called()
                    attention.flash_attn_interface.flash_attn_varlen_func.assert_not_called()
                    self.assertEqual(attention.flash_attn.flash_attn_varlen_func.call_count, 2)


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class StagingContracts(CpuTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(torch.set_grad_enabled, torch.is_grad_enabled())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / "configs").mkdir(parents=True)
        (self.source / "configs/self_forcing_dmd.yaml").write_text("{}")
        (self.source / "model.py").write_text("value = 1\n")
        self.weights, self.vae, self.checkpoint = [self.root / name for name in ("weights.pt", "vae.pth", "override.pt")]
        torch.save({"generator_ema": {"weight": torch.ones(1)}}, self.weights)
        torch.save({"generator": {"module.weight": torch.ones(1)}}, self.checkpoint)
        self.vae.write_bytes(b"vae")
        self.workdir = self.root / "work"
        self.args = SimpleNamespace(source_uri=str(self.source), weights_uri=str(self.weights),
                                    vae_uri=str(self.vae), checkpoint_uri=str(self.checkpoint),
                                    num_frame_per_block=3, local_attn_size=9, sink_size=3,
                                    timestep_shift=5.0, skip_cache_cursor_patch=False,
                                    enable_flash_attn3=False, fa3_import_backend="fa2")
        self.manager = runtime.path_manager()

    def prepare(self):
        omega = ModuleType("omegaconf")
        omega.OmegaConf = SimpleNamespace(
            load=lambda path: Config(model_kwargs=Config()),
            save=lambda config, path: path.write_text(json.dumps(config)))
        with patch.dict(sys.modules, {"omegaconf": omega}), \
                patch.object(runtime, "patch_attention_fallback"), \
                patch.object(runtime, "patch_causal_forward_debug_dict_abi"), \
                patch.object(runtime, "patch_causal_cache_cursor_reads"):
            return runtime.prepare_source(self.args, self.manager, self.workdir)

    def test_directory_reuse_and_effective_checkpoint_audit(self):
        source_root = self.prepare()
        reused, audit = runtime.reuse_prepared_source(self.args, self.manager, self.workdir)
        self.assertEqual(reused, source_root)
        self.assertEqual(audit["contract"]["settings"]["attention_backend"], "fa2")
        self.assertEqual(audit["checkpoint_audit"]["source_key"], "generator")
        state = torch.load(reused / "checkpoints/self_forcing_dmd.pt", map_location="cpu")
        self.assertEqual(list(state["generator_ema"]), ["weight"])
        self.assertEqual(json.loads(json.dumps(audit)), audit)

    def test_zip_reuse_returns_extracted_child(self):
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            for path in self.source.rglob("*"):
                if path.is_file():
                    zipped.write(path, "Self-Forcing-main/" + str(path.relative_to(self.source)))
        self.args.source_uri = str(archive)
        staged = self.prepare()
        reused, _ = runtime.reuse_prepared_source(self.args, self.manager, self.workdir)
        self.assertEqual(staged, reused)
        self.assertEqual(reused, self.workdir / "src/Self-Forcing-main")
        self.assert_reuse_stops_at_loader(reused)

    def reuse_argv(self, *, keep_workdir=True):
        prompts = self.root / "prompts.txt"
        prompts.write_text("test prompt\n")
        return ["unstep", "--fa3-import-backend", "fa2", "--reuse-workdir",
                "--workdir", str(self.root), "--workdir-name", "work",
                "--source-uri", self.args.source_uri, "--weights-uri", self.args.weights_uri,
                "--vae-uri", self.args.vae_uri, "--checkpoint-uri", self.args.checkpoint_uri,
                "--prompt-uri", str(prompts)] + (["--keep-workdir"] if keep_workdir else [])

    def assert_reuse_stops_at_loader(self, expected_root):
        self.addCleanup(torch.set_grad_enabled, torch.is_grad_enabled())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        with patch.object(sys, "argv", self.reuse_argv()), \
                patch.object(runtime, "load_self_forcing_transformer", side_effect=RuntimeError("loader reached")) as load:
            with self.assertRaisesRegex(RuntimeError, "loader reached"):
                runner.main()
        self.assertEqual(load.call_args.kwargs["source_root"], expected_root)

    def test_main_rejects_changed_reuse_before_loading_models(self):
        self.prepare()
        self.vae.write_bytes(b"new asset")
        self.addCleanup(torch.set_grad_enabled, torch.is_grad_enabled())
        with patch.object(sys, "argv", self.reuse_argv(keep_workdir=False)), \
                patch.object(runtime, "load_self_forcing_transformer") as load:
            with self.assertRaisesRegex(RuntimeError, "assets or backend/settings changed"):
                runner.main()
        load.assert_not_called()
        self.assertTrue((self.workdir / runtime.STAGING_AUDIT_FILENAME).is_file())
        self.assertTrue((self.workdir / "src/model.py").is_file())

    def test_main_failed_reuse_preserves_unowned_workdir_and_videos(self):
        self.workdir.mkdir()
        marker = self.workdir / "user_file"
        marker.write_text("preserve")
        videos = self.root / "WRAPPER_GENERATION/videos"
        videos.mkdir(parents=True)
        video = videos / "user_video.mp4"
        video.write_bytes(b"preserve")
        argv = self.reuse_argv(keep_workdir=False) + [
            "--speed-only", "--save-videos", "--video-dir", str(videos)]
        audit = self.workdir / runtime.STAGING_AUDIT_FILENAME
        for contents in (None, "not json", "{}"):
            if contents is not None:
                audit.write_text(contents)
            with self.subTest(audit=contents), patch.object(sys, "argv", argv), \
                    patch.object(runtime, "load_self_forcing_transformer") as load:
                with self.assertRaisesRegex(RuntimeError, "staging audit"):
                    runner.main()
                load.assert_not_called()
            self.assertEqual(marker.read_text(), "preserve")
            self.assertEqual(video.read_bytes(), b"preserve")

    def test_main_prompt_failure_does_not_own_reused_stage(self):
        self.prepare()
        argv = self.reuse_argv(keep_workdir=False) + ["--prompt-indices", "5"]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(RuntimeError, "prompt indices"):
            runner.main()
        self.assertTrue((self.workdir / "src/model.py").is_file())

    def test_main_refuses_fresh_workdir_collisions(self):
        argv = self.reuse_argv(keep_workdir=False)
        argv.remove("--reuse-workdir")
        self.workdir.mkdir()
        for marker in (None, self.workdir / "user_file"):
            if marker is not None:
                marker.write_text("preserve")
            with self.subTest(marker=marker), patch.object(sys, "argv", argv), \
                    patch.object(runtime, "prepare_source") as prepare:
                with self.assertRaisesRegex(RuntimeError, "workdir already exists"):
                    runner.main()
                prepare.assert_not_called()
            self.assertTrue(self.workdir.is_dir())
        self.assertEqual(marker.read_text(), "preserve")

    def test_main_refuses_symlinked_workdir(self):
        argv = self.reuse_argv(keep_workdir=False)
        target = self.root / "unowned"
        target.mkdir()
        (target / "user_file").write_text("preserve")
        for destination in (target, self.root / "missing"):
            self.workdir.symlink_to(destination, target_is_directory=True)
            for reuse in (True, False):
                current = list(argv)
                if not reuse:
                    current.remove("--reuse-workdir")
                with self.subTest(destination=destination, reuse=reuse), \
                        patch.object(sys, "argv", current), self.assertRaisesRegex(RuntimeError, "symlinked workdir"):
                    runner.main()
            self.assertTrue(self.workdir.is_symlink())
            self.workdir.unlink()
        self.assertEqual((target / "user_file").read_text(), "preserve")

    def test_main_protects_repository_and_inputs_directly(self):
        repository = self.root / "repo"
        (repository / "unstep").mkdir(parents=True)
        (repository / "tests").mkdir()
        marker = repository / "unstep/model.py"
        marker.write_text("preserve")
        alias = self.root / "repo_alias"
        alias.symlink_to(repository, target_is_directory=True)
        argv = self.reuse_argv(keep_workdir=False)
        targets = [repository, repository / "unstep", repository / "tests",
                   repository / "unstep/tests", alias / "unstep", self.source,
                   self.source / "new_stage", self.root, self.weights, self.vae,
                   self.checkpoint, self.root / "prompts.txt"]
        for target in targets:
            for reuse in (True, False):
                current = list(argv)
                current[current.index("--workdir") + 1] = str(target.parent)
                current[current.index("--workdir-name") + 1] = target.name
                if not reuse:
                    current.remove("--reuse-workdir")
                with self.subTest(target=target, reuse=reuse), patch.object(runner, "ROOT", repository), \
                        patch.object(sys, "argv", current), patch.object(runtime, "prepare_source") as prepare:
                    with self.assertRaisesRegex(RuntimeError, "protected path"):
                        runner.main()
                    prepare.assert_not_called()
        self.assertEqual(marker.read_text(), "preserve")
        self.assertEqual((self.source / "model.py").read_text(), "value = 1\n")
        for path in (self.weights, self.vae, self.checkpoint, self.root / "prompts.txt"):
            self.assertTrue(path.is_file())
        self.assertFalse((repository / "unstep/tests").exists())
        self.assertFalse((self.source / "new_stage").exists())

    def test_main_cleans_only_owned_workdir_on_failure(self):
        argv = self.reuse_argv(keep_workdir=False)
        argv.remove("--reuse-workdir")
        with patch.object(sys, "argv", argv), \
                patch.object(runtime, "prepare_source", side_effect=RuntimeError("staging failed")):
            with self.assertRaisesRegex(RuntimeError, "staging failed"):
                runner.main()
        self.assertFalse(self.workdir.exists())
        self.prepare()
        videos = self.root / "WRAPPER_GENERATION/videos"
        argv = self.reuse_argv(keep_workdir=False) + [
            "--speed-only", "--save-videos", "--video-dir", str(videos)]
        with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(runtime, "load_self_forcing_transformer", side_effect=RuntimeError("loader reached")):
            with self.assertRaisesRegex(RuntimeError, "loader reached"):
                runner.main()
        self.assertFalse(self.workdir.exists())
        self.assertFalse(videos.exists())

    def test_main_does_not_adopt_existing_speed_video_directory(self):
        self.prepare()
        videos = self.root / "WRAPPER_GENERATION/videos"
        videos.mkdir(parents=True)
        marker = videos / "user_video.mp4"
        marker.write_bytes(b"preserve")
        argv = self.reuse_argv(keep_workdir=False) + [
            "--speed-only", "--save-videos", "--video-dir", str(videos)]
        with patch.object(sys, "argv", argv), \
                patch.object(runtime, "load_self_forcing_transformer") as load:
            with self.assertRaisesRegex(RuntimeError, "video directory already exists"):
                runner.main()
        load.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"preserve")

    def test_reuse_rejects_changed_asset_paths_and_backend(self):
        self.prepare()
        alternate = self.root / "alternate"
        alternate.write_bytes(b"different")
        for name in ("source_uri", "weights_uri", "vae_uri", "checkpoint_uri"):
            args = copy.copy(self.args)
            setattr(args, name, str(alternate))
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "assets or backend/settings changed"):
                runtime.reuse_prepared_source(args, self.manager, self.workdir)
        args = copy.copy(self.args)
        args.fa3_import_backend, args.enable_flash_attn3 = "h100", True
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), \
                self.assertRaisesRegex(RuntimeError, "assets or backend/settings changed"):
            runtime.reuse_prepared_source(args, self.manager, self.workdir)

    def test_reuse_rejects_same_path_asset_changes(self):
        self.prepare()
        self.vae.write_bytes(b"changed vae")
        with self.assertRaisesRegex(RuntimeError, "assets or backend/settings changed"):
            runtime.reuse_prepared_source(self.args, self.manager, self.workdir)

    def test_reuse_rejects_modified_stage_but_ignores_pycache(self):
        source_root = self.prepare()
        cache = source_root / "__pycache__"
        cache.mkdir()
        (cache / "model.pyc").write_bytes(b"generated")
        runtime.reuse_prepared_source(self.args, self.manager, self.workdir)
        (source_root / "model.py").write_text("value = 2\n")
        with self.assertRaisesRegex(RuntimeError, "prepared source changed"):
            runtime.reuse_prepared_source(self.args, self.manager, self.workdir)

    def test_prepare_rejects_directory_symlinks_before_copying(self):
        for name in ("linked_assets", "__pycache__/linked_assets", ".git/linked_assets"):
            link = self.source / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(self.source, target_is_directory=True)
            with self.subTest(name=name), patch.object(runtime, "_copy_source_tree") as copy_source:
                with self.assertRaisesRegex(RuntimeError, "directory symlinks are unsupported"):
                    self.prepare()
                copy_source.assert_not_called()
            self.assertFalse(self.workdir.exists())
            link.unlink()
        alias = self.root / "source_alias"
        alias.symlink_to(self.source, target_is_directory=True)
        self.args.source_uri = str(alias)
        with self.assertRaisesRegex(RuntimeError, "directory symlinks are unsupported"):
            self.prepare()
        self.assertFalse(self.workdir.exists())

    def test_reuse_rejects_linked_directory_edits_and_retargets(self):
        source_root = self.prepare()
        targets = [self.root / "target_a", self.root / "target_b"]
        for target in targets:
            target.mkdir()
            (target / "model.py").write_text("value = 1\n")
        for parent in (self.source, source_root):
            link = parent / "linked_assets"
            link.symlink_to(targets[0], target_is_directory=True)
            for change in ("initial link", "edit target", "retarget"):
                if change == "edit target":
                    (targets[0] / "model.py").write_text("value = 2\n")
                elif change == "retarget":
                    link.unlink()
                    link.symlink_to(targets[1], target_is_directory=True)
                with self.subTest(parent=parent, change=change), \
                        self.assertRaisesRegex(RuntimeError, "directory symlinks are unsupported"):
                    runtime.reuse_prepared_source(self.args, self.manager, self.workdir)
            link.unlink()
            runtime.reuse_prepared_source(self.args, self.manager, self.workdir)

    def test_reuse_rejects_missing_audit_and_escaping_root(self):
        with self.assertRaisesRegex(RuntimeError, "no valid staging audit"):
            runtime.reuse_prepared_source(self.args, self.manager, self.workdir)
        self.prepare()
        audit = runtime.read_staging_audit(self.workdir)
        audit["source_root"] = "../source"
        (self.workdir / runtime.STAGING_AUDIT_FILENAME).write_text(json.dumps(audit))
        with self.assertRaisesRegex(RuntimeError, "outside workdir/src"):
            runtime.reuse_prepared_source(self.args, self.manager, self.workdir)


if __name__ == "__main__":
    unittest.main()
