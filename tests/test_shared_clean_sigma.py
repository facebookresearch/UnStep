# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Check input/output clean sigma against upstream SF without loading a model.

Run: python -m unittest discover -s tests -p 'test_shared_clean_sigma.py' -v
"""

import argparse
import ast
import math
import os
from contextlib import ExitStack
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from unstep import clean_sigma as policy
from unstep import wrapper_runtime as runtime
from unstep.wrapper import OfflineSelfForcingWrapper, WrapperConfig


SF_SOURCE = Path(os.environ.get("UNSTEP_SOURCE_URI", "assets/source/Self-Forcing-main"))
DEVICE = "cpu"


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"missing extracted definitions: {names}")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes,
                        type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class SigmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(DEVICE)
        if cls.device.type == "cpu":
            guard = patch.object(torch.cuda, "_lazy_init",
                                 side_effect=AssertionError("CUDA forbidden in CPU tests"))
            guard.start()
            cls.addClassCleanup(guard.stop)
        cls.scheduler_path = SF_SOURCE / "utils/scheduler.py"
        cls.scheduler_type = extract(cls.scheduler_path, {"FlowMatchScheduler"},
                                     {"torch": torch})["FlowMatchScheduler"]
        cls.runtime = vars(runtime)

    def setUp(self):
        self.scheduler = self.new_scheduler()
        self.transformer = SimpleNamespace(scheduler=self.scheduler)
        self.pipeline = SimpleNamespace(scheduler=self.scheduler, generator=self.transformer)
        self.cfg = SimpleNamespace(clean_sigma_override=0.02)
        self.install_output(self.cfg.clean_sigma_override)

    def new_scheduler(self):
        scheduler = self.scheduler_type(shift=5, sigma_min=0, extra_one_step=True)
        scheduler.set_timesteps(1000, training=True)
        return scheduler

    def install_output(self, value):
        self.runtime["enable_cached_scheduler_sigmas"](
            self.transformer, clean_sigma_override=value)

    def install(self):
        return policy.install_shared_clean_sigma(
            self.pipeline, self.transformer, self.cfg.clean_sigma_override, self.device)

    def rng_state(self):
        states = [torch.get_rng_state()]
        if self.device.type == "cuda":
            states.append(torch.cuda.get_rng_state(self.device))
        return states

    def assert_rng_equal(self, expected):
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, self.rng_state())))

    def test_input_and_existing_output_validation_same_setting(self):
        before = self.rng_state()
        input_result = self.install()
        output_result = self.runtime["validate_clean_sigma_override"](
            self.transformer, clean_sigma_override=self.cfg.clean_sigma_override,
            device=self.device)
        self.assertEqual(input_result, {
            "requested": 0.02, "effective_at_t0": torch.tensor(0.02, dtype=torch.float32).item(),
            "nonzero_timesteps_unchanged": True, "verified": True,
            "scope": "input_renoising_at_input_timestep_zero",
        })
        self.assertEqual(output_result["effective_at_t0"], 0.02)
        self.assertTrue(output_result["verified"])
        self.assertIs(self.pipeline.scheduler, self.transformer.scheduler)
        self.assertEqual(self.scheduler.sigmas.dtype, torch.float32)
        self.assert_rng_equal(before)

    def test_native_statements_plus_exactly_one_where(self):
        native = self.scheduler_type.add_noise
        with patch.object(policy.ast, "fix_missing_locations", wraps=ast.fix_missing_locations) as fix:
            self.install()
        changed = fix.call_args.args[0].body[0]
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "where" for n in ast.walk(changed)), 1)
        changed.body.pop(-3)
        original = ast.parse(policy.dedent(policy.inspect.getsource(native))).body[0]
        self.assertEqual(ast.dump(changed), ast.dump(original))
        self.assertIs(self.scheduler_type.add_noise, native)

    def test_all_table_and_integer_timesteps_ordinary_bit_exact_all_dtypes(self):
        table_before = self.scheduler.timesteps.clone()
        sigma_before = self.scheduler.sigmas.clone()
        self.install()
        native = self.new_scheduler()
        table = table_before.to(self.device)
        times = torch.cat((table, torch.arange(1001, device=self.device)))
        mask = times == 0
        for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                clean = torch.linspace(-2, 2, len(times), device=self.device,
                                       dtype=dtype).reshape(-1, 1, 1, 1)
                noise = clean.flip(0) * 0.25 + 1
                for _ in range(2):
                    expected = native.add_noise(clean, noise, times)
                    actual = self.scheduler.add_noise(clean, noise, times)
                    torch.testing.assert_close(actual[~mask], expected[~mask], rtol=0, atol=0)
                    coefficient = sigma_before.new_full((int(mask.sum()), 1, 1, 1),
                                                       self.cfg.clean_sigma_override,
                                                       device=self.device)
                    clean_expected = ((1 - coefficient) * clean[mask]
                                      + coefficient * noise[mask]).type_as(noise)
                    torch.testing.assert_close(actual[mask], clean_expected, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, dtype)
                baseline_transformer = SimpleNamespace(scheduler=self.scheduler)
                self.runtime["enable_cached_scheduler_sigmas"](
                    baseline_transformer, clean_sigma_override=None)
                output = self.transformer._convert_flow_pred_to_x0(noise, clean, times)
                baseline = baseline_transformer._convert_flow_pred_to_x0(noise, clean, times)
                torch.testing.assert_close(output[~mask], baseline[~mask], rtol=0, atol=0)
                torch.testing.assert_close(output[mask],
                    (clean[mask].double() - self.cfg.clean_sigma_override
                     * noise[mask].double()).to(dtype), rtol=0, atol=0)
        torch.testing.assert_close(self.scheduler.timesteps.cpu(), table_before, rtol=0, atol=0)
        torch.testing.assert_close(self.scheduler.sigmas.cpu(), sigma_before, rtol=0, atol=0)
        self.assertEqual(self.scheduler.sigmas.device, self.device)

    def test_two_dimensional_mixed_timesteps_and_sample_coefficient(self):
        self.install()
        times = torch.tensor([[0, 1, 4], [625, 1000, 0]], device=self.device)
        ones = torch.ones((6, 2, 2, 2), dtype=torch.float32, device=self.device)
        zeros = torch.zeros_like(ones)
        native = self.new_scheduler()
        expected = native.add_noise(ones, zeros, times)
        mask = times.flatten() == 0
        expected[mask] = 1 - torch.tensor(0.02, dtype=torch.float32, device=self.device)
        actual = self.scheduler.add_noise(ones, zeros, times)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, ones.shape)
        self.assertEqual(times.tolist(), [[0, 1, 4], [625, 1000, 0]])

    def test_single_transition_draw_global_and_explicit_generator(self):
        self.install()
        original_draw = self.runtime["randn_like_with_generator"]
        clean = torch.ones((1, 3, 2, 2, 2), dtype=torch.bfloat16, device=self.device)
        flat = clean.flatten(0, 1)
        for explicit in (False, True):
            for timestep in (0, 625):
                with self.subTest(explicit=explicit, timestep=timestep):
                    generator = torch.Generator(device=self.device).manual_seed(17) if explicit else None
                    state = generator.get_state() if explicit else (
                        torch.cuda.get_rng_state(self.device) if self.device.type == "cuda"
                        else torch.get_rng_state())
                    noise = original_draw(flat, generator)
                    expected_state = generator.get_state() if explicit else (
                        torch.cuda.get_rng_state(self.device) if self.device.type == "cuda"
                        else torch.get_rng_state())
                    if explicit:
                        generator.set_state(state)
                    elif self.device.type == "cuda":
                        torch.cuda.set_rng_state(state, self.device)
                    else:
                        torch.set_rng_state(state)
                    with patch.dict(self.runtime, {"randn_like_with_generator":
                            unittest.mock.Mock(wraps=original_draw)}):
                        actual = self.runtime["add_noise_transition"](
                            pipeline=self.pipeline, denoised_pred=clean,
                            next_timestep=timestep, generator=generator)
                        self.assertEqual(self.runtime["randn_like_with_generator"].call_count, 1)
                    after = generator.get_state() if explicit else (
                        torch.cuda.get_rng_state(self.device) if self.device.type == "cuda"
                        else torch.get_rng_state())
                    self.assertTrue(torch.equal(after, expected_state))
                    times = torch.full((3,), timestep, device=self.device, dtype=torch.long)
                    expected = self.scheduler.add_noise(flat, noise, times).unflatten(0, clean.shape[:2])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_sigma_identity_shift_and_precision_fail_closed(self):
        original = self.scheduler.add_noise
        for value in (None, True, -0.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.install_shared_clean_sigma(self.pipeline, self.transformer, value, self.device)
        other = SimpleNamespace(scheduler=self.scheduler)
        with self.assertRaisesRegex(RuntimeError, "identity"):
            policy.install_shared_clean_sigma(self.pipeline, other, 0.02, self.device)
        self.transformer.scheduler = self.new_scheduler()
        with self.assertRaisesRegex(RuntimeError, "identity"):
            self.install()
        self.transformer.scheduler = self.scheduler
        self.scheduler.shift = 8
        with self.assertRaisesRegex(RuntimeError, "shift 5"):
            self.install()
        self.scheduler.shift = 5
        self.scheduler.sigmas = self.scheduler.sigmas.double()
        with self.assertRaisesRegex(RuntimeError, "FP32"):
            self.install()
        self.assertEqual(self.scheduler.add_noise, original)

    def test_source_drift_and_reinstallation_fail_closed(self):
        text = policy.dedent(policy.inspect.getsource(self.scheduler.add_noise))
        original = self.scheduler.add_noise
        for changed in (text.replace("1 - sigma", "1 + sigma"),
                        text.replace("sigma = self.sigmas", "sigma = self.sigmas * 2 #"),
                        "def unexpected():\n    pass\n" + text):
            with self.subTest(changed=changed), patch.object(policy.inspect, "getsource", return_value=changed):
                with self.assertRaises(RuntimeError):
                    self.install()
                self.assertEqual(self.scheduler.add_noise, original)
        self.install()
        with self.assertRaises(RuntimeError):
            self.install()

    def test_input_and_output_mismatches_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "coefficient mismatch"):
            policy._validate_input(self.scheduler, 0.02, self.device)
        self.install()
        with self.assertRaisesRegex(RuntimeError, "coefficient mismatch"):
            policy._validate_input(self.scheduler, 0.025, self.device)
        self.install_output(0.025)
        with self.assertRaisesRegex(RuntimeError, "clean sigma validation failed"):
            self.runtime["validate_clean_sigma_override"](
                self.transformer, clean_sigma_override=self.cfg.clean_sigma_override,
                device=self.device)

    def test_validator_detects_nonzero_corruption_and_rng_draw(self):
        self.install()
        correct = self.scheduler.add_noise
        def corrupt(scheduler, clean, noise, times):
            result = correct(clean, noise, times)
            result[0] += 1
            return result
        self.scheduler.add_noise = MethodType(corrupt, self.scheduler)
        with self.assertRaisesRegex(RuntimeError, "coefficient mismatch"):
            policy._validate_input(self.scheduler, 0.02, self.device)
        def consumes_rng(scheduler, clean, noise, times):
            torch.rand(1, device=self.device)
            return correct(clean, noise, times)
        self.scheduler.add_noise = MethodType(consumes_rng, self.scheduler)
        with self.assertRaisesRegex(RuntimeError, "RNG state"):
            policy._validate_input(self.scheduler, 0.02, self.device)

    def test_failed_validation_restores_native_method(self):
        original = self.scheduler.add_noise
        with patch.object(policy, "_validate_input", side_effect=RuntimeError("diagnostic failure")):
            with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
                self.install()
        self.assertEqual(self.scheduler.add_noise, original)

    def test_wrapper_installs_and_uses_same_sigma_for_input_and_emission(self):
        class Transformer:
            def __init__(self, scheduler):
                self.scheduler = scheduler
                self.model = SimpleNamespace(blocks=[])
                self.calls = []

            def compile(self, **kwargs):
                pass

            def __call__(self, *, noisy_image_or_video, timestep, **kwargs):
                self.calls.append((noisy_image_or_video.clone(), timestep.clone()))
                flat = noisy_image_or_video.flatten(0, 1)
                flow = torch.full_like(flat, 2)
                clean = self._convert_flow_pred_to_x0(flow, flat, timestep.flatten())
                return flow, clean.unflatten(0, noisy_image_or_video.shape[:2])

        transformer = Transformer(self.scheduler)
        pipeline = SimpleNamespace(
            scheduler=self.scheduler, generator=transformer,
            frame_seq_length=runtime.WAN_TOKENS_PER_LATENT_FRAME,
            vae=SimpleNamespace(to=lambda **kwargs: None,
                                model=SimpleNamespace(_sf_compiled_decode=True)),
        )
        wrapper = OfflineSelfForcingWrapper()
        self.assertEqual(wrapper.config.clean_sigma_override, 0.02)
        unrelated_runtime = (
            "apply_layerwise_local_attention", "install_flash_attention_seqlen_cache",
            "prewarm_flash_attention_seqlen_cache", "install_causal_rope_triton_fp32_cache",
            "enable_fast_vae_cache_clear", "enable_vae_decode_runtime_fast_path",
            "enable_cached_vae_list_cat_output", "enable_vae_channels_last_3d",
            "enable_cached_vae_block_decode",
        )
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark",
                        torch.backends.cudnn.benchmark)
        with ExitStack() as stack:
            for name in unrelated_runtime:
                stack.enter_context(patch.object(runtime, name))
            wrapper.install_runtime(pipeline=pipeline, target_transformer=transformer,
                                    method_transformer=transformer, device=self.device)
        self.assertTrue(wrapper.clean_sigma_input_validation["verified"])
        result = runtime.validate_clean_sigma_override(
            transformer, clean_sigma_override=wrapper.config.clean_sigma_override,
            device=self.device)
        self.assertEqual(result["effective_at_t0"], 0.02)

        initial = torch.ones((1, 3, 2, 2, 2), device=self.device)
        seed = 17
        expected_noise = torch.randn(initial.flatten(0, 1).shape, device=self.device,
                                    generator=torch.Generator(device=self.device).manual_seed(seed))
        # At t=1000 the first clean prediction is 1 - 1*2 = -1.
        sigma = torch.tensor(0.02, dtype=torch.float32, device=self.device)
        expected_input = ((1 - sigma) * -torch.ones_like(expected_noise)
                          + sigma * expected_noise).unflatten(0, initial.shape[:2])
        expected_output = (expected_input.double() - 0.02 * 2).float()
        with patch.object(runtime, "randn_like_with_generator",
                          wraps=runtime.randn_like_with_generator) as draw:
            output, count = wrapper.denoise_chunk(
                transformer=transformer, pipeline=pipeline, noisy_input=initial,
                conditional_dict={}, kv_cache=[], crossattn_cache=[],
                current_start_frame=0, block_idx=0, num_blocks=1,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                timesteps_override=[1000.0, 0.0],
            )
        self.assertEqual(count, 2)
        self.assertEqual(draw.call_count, 1)
        torch.testing.assert_close(transformer.calls[-1][0], expected_input, rtol=0, atol=0)
        self.assertTrue(torch.all(transformer.calls[-1][1] == 0))
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf-source", type=Path, default=SF_SOURCE)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0", "cuda:1"))
    args, remaining = parser.parse_known_args()
    SF_SOURCE, DEVICE = args.sf_source, args.device
    torch.set_num_threads(1)
    unittest.main(argv=[__file__, *remaining])
