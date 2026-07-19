"""Optional-stopping stress test (RQ3, Sec. 12.6).

Compares empirical type-I error (falsely certifying non-inferiority of a
materially regressed anchor) under adversarial peeking:

  - fixed-sample normal (CLT) test evaluated repeatedly  -> inflates
  - e-process gate                                       -> controlled
"""
import numpy as np
from spectra_rsi.gate import EProcess


def clt_repeated_test(stream, tau, alpha):
    """Naive: recompute a one-sided z-test after every batch, stop at first
    'significant' favorable result (the invalid-but-common practice)."""
    from math import sqrt
    n, s, ss = 0, 0.0, 0.0
    for batch in stream:
        for d in batch:
            n += 1
            s += d
            ss += d * d
        if n > 30:
            mean = s / n
            var = max(ss / n - mean ** 2, 1e-12)
            z = (mean + tau) / sqrt(var / n)
            if z > 1.645:                       # nominal alpha = 0.05
                return True
    return False


def eprocess_test(stream, tau, alpha):
    ep = EProcess(tau=tau, alpha=alpha, bound=1.0)
    for batch in stream:
        ep.update(batch)
        if ep.rejects_null:
            return True
    return False


def main():
    tau, alpha = 0.05, 0.05
    n_sims, n_batches, batch = 500, 60, 25
    rng = np.random.default_rng(0)
    clt_fr, ep_fr = 0, 0
    for _ in range(n_sims):
        # TRUE null: anchor sits exactly at the material-regression boundary
        stream = [np.clip(rng.normal(-tau, 0.30, batch), -1, 1)
                  for _ in range(n_batches)]
        clt_fr += clt_repeated_test(iter(stream), tau, alpha)
        ep_fr += eprocess_test(iter(stream), tau, alpha)
    print(f"simulations                  : {n_sims}")
    print(f"nominal alpha                : {alpha:.3f}")
    print(f"CLT repeated-test type-I err : {clt_fr / n_sims:.3f}   <-- inflated")
    print(f"e-process gate type-I err    : {ep_fr / n_sims:.3f}   <-- controlled")


if __name__ == "__main__":
    main()
