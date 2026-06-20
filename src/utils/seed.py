import numpy as np, random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def make_run_id(method: str, dataset: str):
    import datetime as dt
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{method}_{dataset}"
