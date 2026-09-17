"""Check clean-sigma arithmetic against the upstream shift-five scheduler."""

import argparse
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from unstep import wrapper_runtime as runtime


ARGS = SimpleNamespace(
    sf_source=Path(os.environ.get("UNSTEP_SOURCE_URI", "assets/source/Self-Forcing-main")),
    device="cpu",
)


class CleanSigmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "sf_scheduler", ARGS.sf_source / "utils/scheduler.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.scheduler_type = module.FlowMatchScheduler
        cls.device = torch.device(ARGS.device)

    def setUp(self):
        self.scheduler = self.scheduler_type(shift=5, sigma_min=0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.assertGreater(self.scheduler.timesteps.min().item(), 0)

    def converter(self, requested):
        transformer = SimpleNamespace(scheduler=self.scheduler)
        runtime.enable_cached_scheduler_sigmas(transformer, clean_sigma_override=requested)
        return transformer._convert_flow_pred_to_x0

    def expected(self, times, requested):
        table = self.scheduler.timesteps.double().to(self.device)
        indices = (table[None, :] - times[:, None]).abs().argmin(dim=1)
        values = self.scheduler.sigmas.double().to(self.device)[indices]
        if requested is not None:
            values[times == 0] = requested
        return values[:, None, None, None]

    def test_override_uses_input_zero_not_nearest_table_entry(self):
        # Nonzero 1 and 4 also resolve to the last table entry: never override them.
        times = torch.tensor([0, 1, 4, 625, 1000, 0], device=self.device)
        flow = torch.ones((len(times), 1, 1, 1), dtype=torch.float64, device=self.device)
        xt = torch.zeros_like(flow)
        native = self.scheduler.sigmas[-1].item()
        for requested in (None, 0.0, native, 0.001, 0.005, 0.007, 0.01, 0.02, 0.025, 0.03):
            with self.subTest(sigma=requested):
                convert = self.converter(requested)
                for _ in range(2):
                    torch.testing.assert_close(
                        -convert(flow, xt, times), self.expected(times, requested),
                        rtol=0, atol=0,
                    )

    def test_real_dtypes_and_all_nonzero_steps(self):
        times = torch.arange(1001, device=self.device)
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            flow = torch.linspace(-2, 2, 1001, device=self.device, dtype=dtype)[:, None, None, None]
            xt = flow.flip(0) * 0.5
            default = self.converter(None)(flow, xt, times)
            native = self.scheduler.sigmas[-1].item()
            exact = self.converter(native)(flow, xt, times)
            torch.testing.assert_close(default, exact, rtol=0, atol=0)
            actual = self.converter(0.02)(flow, xt, times)
            expected = (xt.double() - self.expected(times, 0.02) * flow.double()).to(dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual[1:], default[1:], rtol=0, atol=0)
            self.assertEqual(actual.dtype, dtype)

    def test_scheduler_and_input_renoising_are_unchanged(self):
        before_steps = self.scheduler.timesteps.clone()
        before_sigmas = self.scheduler.sigmas.clone()
        times = torch.tensor([0, 625, 1000])
        xt = torch.zeros((3, 1, 1, 1))
        noise = torch.ones_like(xt)
        before_noise = self.scheduler.add_noise(xt, noise, times)
        self.converter(0.02)
        torch.testing.assert_close(self.scheduler.timesteps, before_steps, rtol=0, atol=0)
        torch.testing.assert_close(self.scheduler.sigmas, before_sigmas, rtol=0, atol=0)
        torch.testing.assert_close(
            self.scheduler.add_noise(xt, noise, times), before_noise, rtol=0, atol=0
        )
        self.assertEqual(self.scheduler.shift, 5)

    def test_startup_validation_rejects_ignored_override(self):
        transformer = SimpleNamespace(scheduler=self.scheduler)
        runtime.enable_cached_scheduler_sigmas(transformer, clean_sigma_override=None)
        with self.assertRaisesRegex(RuntimeError, "clean sigma validation failed"):
            runtime.validate_clean_sigma_override(
                transformer, clean_sigma_override=0.02, device=self.device
            )
        for requested in (None, 0.005, 0.01, 0.02, None):
            runtime.enable_cached_scheduler_sigmas(transformer, clean_sigma_override=requested)
            cpu_rng = torch.get_rng_state()
            device_rng = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            result = runtime.validate_clean_sigma_override(
                transformer, clean_sigma_override=requested, device=self.device
            )
            self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
            if device_rng is not None:
                self.assertTrue(torch.equal(device_rng, torch.cuda.get_rng_state(self.device)))
            expected = self.scheduler.sigmas[-1].item() if requested is None else requested
            self.assertEqual(result["effective_at_t0"], expected)
            self.assertTrue(result["verified"])

    def test_invalid_overrides_rejected(self):
        for requested in (-0.001, float("nan"), float("inf"), float("-inf")):
            with self.subTest(sigma=requested), self.assertRaises(ValueError):
                self.converter(requested)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_compiled_converter(self):
        if self.device.type != "cuda":
            self.skipTest("Run with --device cuda to check torch.compile")
        times = torch.tensor([0, 625, 0], device=self.device)
        flow = torch.ones((3, 2, 2, 2), dtype=torch.bfloat16, device=self.device)
        xt = torch.zeros_like(flow)
        for requested in (None, 0.005, 0.01, 0.02):
            convert = self.converter(requested)
            expected = convert(flow, xt, times)
            compiled = torch.compile(convert, mode="max-autotune-no-cudagraphs", fullgraph=True)
            torch.testing.assert_close(compiled(flow, xt, times), expected, rtol=0, atol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf-source", type=Path, default=Path(os.environ.get(
        "UNSTEP_SOURCE_URI", "assets/source/Self-Forcing-main"
    )))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ARGS, remaining = parser.parse_known_args()
    unittest.main(argv=[__file__, *remaining])
