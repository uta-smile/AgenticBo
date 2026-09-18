"""Exercise the real upstream sampling loop with a cheap synthetic denoiser.

These verify adapter mechanics, not protein quality or checkpoint determinism.
"""
import random

import numpy as np
import pytest
import torch
from boltz.model.modules.diffusionv2 import AtomDiffusion

from generator.native_noise import NativeNoise, fixed_randomness, inject_native_noise, sampler_source_audit
from latent.inspect import inspect_native


class SamplerHarness:
    sample = AtomDiffusion.sample
    device = torch.device("cpu")
    training = False
    num_sampling_steps = 3
    gamma_min = 1.0
    gamma_0 = 0.8
    step_scale = 1.5
    noise_scale = 1.003
    alignment_reverse_diff = False

    def sample_schedule(self, steps):
        return torch.tensor([4.0, 2.0, 0.5, 0.0])

    def preconditioned_network_forward(self, coordinates, sigma, **kwargs):
        return coordinates * 0.8


STEERING = dict(fk_steering=False, physical_guidance_update=False, contact_guidance_update=False)


def sample(harness, seed=19):
    with fixed_randomness(seed):
        return harness.sample(torch.ones(1, 32), steering_args=STEERING)["sample_atom_coords"]


def test_injection_matches_unmodified_native_draw_and_preserves_rng():
    harness = SamplerHarness()
    with fixed_randomness(19):
        z = torch.randn(1, 32, 3)
    original = sample(harness)
    with inject_native_noise(harness, z) as injection:
        injected = sample(harness)
    assert injection.calls == 1
    assert torch.equal(original, injected)
    assert "sample" not in harness.__dict__


def test_different_latents_are_used_and_repeats_deterministic():
    harness = SamplerHarness()
    z = torch.randn(1, 32, 3)
    results = []
    for latent in [z, z, z + torch.randn_like(z) * 0.1]:
        with inject_native_noise(harness, latent):
            results.append(sample(harness))
    assert torch.equal(results[0], results[1])
    assert not torch.equal(results[0], results[2])
    assert torch.isfinite(results[2]).all()


def test_every_coordinate_including_padding_reaches_initialization():
    z = torch.arange(96, dtype=torch.float32).reshape(1, 32, 3)
    injection = NativeNoise(z)
    assert torch.equal(injection(torch.randn_like(z)), z)
    with pytest.raises(RuntimeError, match="one diffusion sample"):
        injection(torch.randn_like(z))


def test_bad_shapes_and_nonfinite_values_fail():
    with pytest.raises(ValueError, match="shape"):
        NativeNoise(torch.ones(96))
    with pytest.raises(ValueError, match="finite"):
        NativeNoise(torch.full((1, 32, 3), float("nan")))
    with pytest.raises(ValueError, match="differs"):
        NativeNoise(torch.ones(1, 32, 3))(torch.ones(1, 64, 3))


def test_restores_sampler_after_failure_and_rejects_unused_latent():
    harness = SamplerHarness()
    with pytest.raises(ValueError, match="failure"):
        with inject_native_noise(harness, torch.zeros(1, 32, 3)):
            raise ValueError("failure")
    assert "sample" not in harness.__dict__
    with pytest.raises(RuntimeError, match="exactly one"):
        with inject_native_noise(harness, torch.zeros(1, 32, 3)):
            pass
    assert "sample" not in harness.__dict__


def test_fixed_randomness_restores_caller_state_after_failure():
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    mode = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(ValueError):
        with fixed_randomness(2):
            random.random(), np.random.rand(), torch.randn(3)
            raise ValueError("failure")
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.are_deterministic_algorithms_enabled() == mode


def test_report_uses_measured_full_shape():
    mask = torch.cat([torch.ones(1, 29), torch.zeros(1, 3)], dim=1)
    report = inspect_native(torch.randn(1, 32, 3), mask)
    assert report["flattened_D"] == 96
    assert report["native_latent_shape"] == [1, 32, 3]
    assert report["active_atoms"] == 29
    assert report["generation_deterministic_adapter"] is None
    assert report["distribution"].startswith("independent standard Normal")
    assert sampler_source_audit(AtomDiffusion.sample)["initialization_matches"] == 1
