"""Inject native initial noise and provide a separate deterministic sampler mode.

The default source transformation is deliberately narrow: one expression
changes. The PF-ODE experiment is a separate source-pinned transformation that
also removes stochastic sampler terms. No global torch function is monkey-
patched. A private function-global dictionary keeps adapter instances isolated.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from contextlib import contextmanager
from importlib.metadata import version

import torch

SUPPORTED_BOLTZ = "2.2.1"
SUPPORTED_SAMPLE_SHA256 = "f354e2ad65affa4a21f7cad5e3c24e79245b17d582b6e3edfe4b75377d960dfe"
INITIAL_EXPRESSION = "init_sigma * torch.randn(shape, device=self.device)"


def _assignment_names(node):
    names = []
    for target in node.targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            names.extend(item.id for item in target.elts if isinstance(item, ast.Name))
    return set(names)


def sampler_source_audit(sample) -> dict:
    source = textwrap.dedent(inspect.getsource(sample))
    tree = ast.parse(source)
    expected = ast.dump(ast.parse(INITIAL_EXPRESSION, mode="eval").body)
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "atom_coords" for t in node.targets)
               and ast.dump(node.value) == expected]
    if len(matches) != 1:
        raise RuntimeError("Boltz sampler initialization changed; review adapter before running")
    return {"sample_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "initialization_matches": len(matches), "expression": INITIAL_EXPRESSION}


def compile_injected_sampler(sample, initial_noise):
    """Return an isolated sampler callable; initial_noise receives its native draw."""
    sampler_source_audit(sample)
    tree = ast.parse(textwrap.dedent(inspect.getsource(sample)))
    function = tree.body[0]
    function.decorator_list = []
    expected = ast.dump(ast.parse(INITIAL_EXPRESSION, mode="eval").body)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "atom_coords" for t in node.targets)
                and ast.dump(node.value) == expected):
            # Consume the original draw even on injection, preserving later RNG state.
            node.value.right = ast.Call(func=ast.Name(id="_native_initial_noise", ctx=ast.Load()),
                                        args=[node.value.right], keywords=[])
    ast.fix_missing_locations(tree)
    namespace = dict(sample.__globals__)
    namespace["_native_initial_noise"] = initial_noise
    exec(compile(tree, "<boltz2-native-noise>", "exec"), namespace)
    return namespace[function.name]


def compile_pf_ode_sampler(sample, initial_noise):
    """Compile a deterministic Euler probability-flow variant of Boltz's sampler.

    This is intentionally a source-pinned transformation of the installed
    sampler rather than a second denoising implementation. It keeps Boltz's
    network, schedule, conditioning, and denoising update, while making the
    following sampler terms deterministic:

    * the supplied native latent is the initial state;
    * churn is disabled (gamma = 0);
    * transition noise is zero;
    * coordinate augmentation is the identity transform; and
    * the Euler step scale is supplied by the adapter as one.

    The resulting path is the unit-step Euler discretization of Boltz's
    denoiser-based probability-flow update. It is a separate experimental
    mode and does not alter the default stochastic sampler.
    """
    sampler_source_audit(sample)
    tree = ast.parse(textwrap.dedent(inspect.getsource(sample)))
    function = tree.body[0]
    function.decorator_list = []

    replaced = {"initial": 0, "gammas": 0, "augmentation": 0, "eps": 0}
    expected_initial = ast.dump(ast.parse(INITIAL_EXPRESSION, mode="eval").body)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = _assignment_names(node)
        if names == {"atom_coords"} and ast.dump(node.value) == expected_initial:
            # Do not draw or consume a random initial tensor in PF-ODE mode.
            # A zero reference tensor supplies device, dtype, and shape to the
            # existing NativeNoise validator.
            node.value = ast.parse(
                "init_sigma * _native_initial_noise(torch.zeros(shape, device=self.device))",
                mode="eval",
            ).body
            replaced["initial"] += 1
        elif names == {"gammas"}:
            node.value = ast.parse("torch.zeros_like(sigmas)", mode="eval").body
            replaced["gammas"] += 1
        elif names == {"random_R", "random_tr"}:
            node.value = ast.parse(
                "_deterministic_augmentation(multiplicity, device=atom_coords.device, dtype=atom_coords.dtype)",
                mode="eval",
            ).body
            replaced["augmentation"] += 1
        elif names == {"eps"}:
            node.value = ast.parse("torch.zeros_like(atom_coords)", mode="eval").body
            replaced["eps"] += 1

    if replaced != {"initial": 1, "gammas": 1, "augmentation": 1, "eps": 1}:
        raise RuntimeError(f"Unexpected Boltz sampler structure for PF-ODE transformation: {replaced}")

    ast.fix_missing_locations(tree)
    namespace = dict(sample.__globals__)
    namespace["_native_initial_noise"] = initial_noise
    namespace["_deterministic_augmentation"] = deterministic_augmentation
    exec(compile(tree, "<boltz2-pf-ode>", "exec"), namespace)
    return namespace[function.name]


class NativeNoise:
    def __init__(self, latent: torch.Tensor):
        if latent.ndim != 3 or latent.shape[0] != 1 or latent.shape[-1] != 3 or latent.shape[1] < 1:
            raise ValueError("Expected full native shape [1, padded_atoms, 3]")
        if not latent.is_floating_point() or not torch.isfinite(latent).all():
            raise ValueError("Native noise must be a finite floating tensor")
        self.latent = latent.detach().clone()
        self.calls = 0

    def __call__(self, native_draw: torch.Tensor) -> torch.Tensor:
        if self.calls:
            raise RuntimeError("Only one diffusion sample is allowed per expensive evaluation")
        if native_draw.shape != self.latent.shape:
            raise ValueError(f"Native shape {tuple(native_draw.shape)} differs from latent {tuple(self.latent.shape)}")
        converted = self.latent.to(native_draw)
        if not torch.isfinite(converted).all():
            raise ValueError("Native noise overflow after dtype conversion")
        self.calls += 1
        return converted


def deterministic_augmentation(multiplicity, *, device, dtype):
    """Return the identity rotation and zero translation for PF-ODE sampling."""
    rotation = torch.eye(3, device=device, dtype=dtype).expand(multiplicity, -1, -1)
    translation = torch.zeros((multiplicity, 1, 3), device=device, dtype=dtype)
    return rotation, translation


@contextmanager
def inject_native_noise(structure_module, latent: torch.Tensor):
    """Temporarily replace this module's sampler, and restore it on every exit."""
    import types

    if version("boltz") != SUPPORTED_BOLTZ:
        raise RuntimeError(f"Expected boltz=={SUPPORTED_BOLTZ}")
    original = structure_module.sample
    if sampler_source_audit(original)["sample_source_sha256"] != SUPPORTED_SAMPLE_SHA256:
        raise RuntimeError("Installed sampler differs from the audited Boltz 2.2.1 source")
    had_override = "sample" in structure_module.__dict__
    injection = NativeNoise(latent)
    function = compile_injected_sampler(original.__func__, injection)
    structure_module.sample = types.MethodType(function, structure_module)
    try:
        yield injection
        if injection.calls != 1:
            raise RuntimeError("Generator did not consume exactly one native latent")
    finally:
        if had_override:
            structure_module.sample = original
        else:
            del structure_module.sample


@contextmanager
def inject_pf_ode_noise(structure_module, latent: torch.Tensor):
    """Temporarily replace Boltz sampling with the deterministic PF-ODE path."""
    import types

    if version("boltz") != SUPPORTED_BOLTZ:
        raise RuntimeError(f"Expected boltz=={SUPPORTED_BOLTZ}")
    original = structure_module.sample
    if sampler_source_audit(original)["sample_source_sha256"] != SUPPORTED_SAMPLE_SHA256:
        raise RuntimeError("Installed sampler differs from the audited Boltz 2.2.1 source")
    had_override = "sample" in structure_module.__dict__
    injection = NativeNoise(latent)
    function = compile_pf_ode_sampler(original.__func__, injection)
    structure_module.sample = types.MethodType(function, structure_module)
    try:
        yield injection
        if injection.calls != 1:
            raise RuntimeError("Generator did not consume exactly one native latent")
    finally:
        if had_override:
            structure_module.sample = original
        else:
            del structure_module.sample


@contextmanager
def fixed_randomness(seed: int):
    """Freeze all runtime randomness for G(z), then restore caller RNG states."""
    import random
    import numpy as np

    py_state, np_state = random.getstate(), np.random.get_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    with torch.random.fork_rng():
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cudnn.benchmark = cudnn_benchmark
