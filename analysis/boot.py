"""Task-clustered bootstrap helpers (resample tasks with replacement), vectorised and seeded."""
import numpy as np

BOOT, SEED = 2000, 0


def task_counts(tasks, B=BOOT, keys=None):
    """-> (keys, inv, counts): counts[b, t] = how often task t is drawn in resample b.
    Pass `keys` (all task ids) to get the same resamples for subsets that lack some tasks."""
    keys = np.unique(tasks if keys is None else keys)
    inv = np.searchsorted(keys, tasks)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(keys), (B, len(keys)))
    counts = np.stack([np.bincount(d, minlength=len(keys)) for d in draws])
    return keys, inv, counts


def auc_point_and_draws(y, pred, tasks, B=BOOT, keys=None):
    """AUROC (ties count half) and its bootstrap draws, one weighted AUROC per resample."""
    _, inv, counts = task_counts(tasks, B, keys)
    W = counts[:, inv].astype(float)
    pos, neg = y == 1, y == 0
    S = (pred[pos, None] > pred[None, neg]) + 0.5 * (pred[pos, None] == pred[None, neg])
    wp, wn = W[:, pos], W[:, neg]
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = ((wp @ S) * wn).sum(1) / (wp.sum(1) * wn.sum(1))
    return S.sum() / (pos.sum() * neg.sum()), draws


def ci(draws):
    return np.nanpercentile(draws, [2.5, 97.5])


def task_sums(tasks, *cols, keys=None):
    """Per-task sums of each column, aligned with task_counts' keys."""
    keys = np.unique(tasks if keys is None else keys)
    inv = np.searchsorted(keys, tasks)
    return [np.bincount(inv, weights=np.asarray(c, dtype=float), minlength=len(keys)) for c in cols]
