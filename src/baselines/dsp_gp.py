from dsp.tools import OptimizationBackend


def build_backend(*args, **kwargs):
    return OptimizationBackend(*args, **kwargs, dsp=True)
