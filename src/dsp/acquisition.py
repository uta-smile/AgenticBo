from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound


def acquisition(model, best_f: float, *, name: str = "log_ei", beta: float = 2.0):
    if name == "log_ei":
        return LogExpectedImprovement(model, best_f=best_f, maximize=True)
    if name == "ucb":
        if not 0 < beta < float("inf"):
            raise ValueError("UCB beta must be finite and positive")
        return UpperConfidenceBound(model, beta=beta, maximize=True)
    raise ValueError("Acquisition must be log_ei or ucb")
