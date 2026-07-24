"""
Reference: Big Chalk's constrained-fit approach (Alex Hathcock, shared in the
2026-07-06 sponsor review meeting chat). Preserved verbatim below, plus an
equivalence test against our engine.

Alex fits the marketing-mix model with scipy.optimize.curve_fit (method TRF)
on a hand-built linear function, with coefficient bounds assembled per GROUP
(intercept / controls / positives / negatives / media) in fixed column order.

Our engine (capstone_pipeline.constrained_fit) uses scipy.optimize.lsq_linear
(also TRF) with bounds per VARIABLE from variable_config.csv. For a linear
model the two are the same optimization; curve_fit is the general nonlinear
wrapper, lsq_linear the direct linear solver. Differences that matter to us:

  - per-variable bounds subsume per-group bounds (a group bound is just the
    same [lo, hi] repeated), and cannot misalign if column order changes;
  - lsq_linear needs no p0 initial guess;
  - verified numerically equivalent on the REPORTED model (full window): the
    largest coefficient gap is ~2e-08 of the largest coefficient's scale on
    Brand 1 x Channel 1 (run this file to reproduce).

Conclusion (for the methodology write-up): we keep lsq_linear; results match
the sponsor's production approach to floating-point precision.
"""
import numpy as np


# --------------------- Alex's code, verbatim ---------------------

def linear_function_var(data, a, *coeffs):
    """
    Returns the right side of a linear funtion (in {} to the right) with the form f(data) = { a + b1*data[:,0] + b2*data[:,1]...bn*data[:,n+1]}
    """
    #returns right side of linear function with the form f(data) = (starts here->) a + b1*data[:,0] + b2*data[:,1]...bn*data[:,n+1]
    coeffs = np.array(coeffs)
    assert data.shape[1] == len(coeffs), \
        f"The number of coefficients ({len(coeffs)}) must match the number of variables ({data.shape[1]})."

    # Compute the linear function
    return a + np.dot(data, coeffs)


def prepare_bounds_custom(intercept_value,
                         control_length, positive_control_length, negative_control_length, media_length,
                         control_value, positive_control_value, negative_control_value, media_value
                         ):
    """

    """
    bounds_array = []
    bounds_array.append(intercept_value)
    for i in range(control_length):
        bounds_array.append(control_value)
    for i in range(positive_control_length):
        bounds_array.append(positive_control_value)
    for i in range(negative_control_length):
        bounds_array.append(negative_control_value)

    for i in range(media_length):
        bounds_array.append(media_value)

    return bounds_array


# --------------------- equivalence test ---------------------

if __name__ == "__main__":
    import os
    import warnings; warnings.filterwarnings("ignore")
    from scipy.optimize import curve_fit
    import capstone_pipeline as cp

    # resolve the data path relative to THIS file so it runs from any cwd
    _data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "Anonymized Data for Project.xlsx")
    r = cp.run_slice(_data, "Brand 1", "Channel 1")
    # Compare on the SAME data the reported model is fit on — the full reported
    # window (all weeks), not the validation training split.
    X = r["X_all"][r["selected"]]
    y = r["y"]
    lo, hi = cp._bounds_from_specs(r["specs_by_name"], list(X.columns))

    p0 = np.clip(np.zeros(len(lo)), lo, np.where(np.isinf(hi), 0, hi))
    popt, _ = curve_fit(linear_function_var, X.values, y.values, p0=p0,
                        bounds=(lo, hi), method="trf", maxfev=5000)

    ours = r["fit"].coef.values
    # Robust to coefficients pinned at ~0: measure the largest absolute
    # coefficient gap relative to the LARGEST coefficient's scale (a plain
    # per-coef relative diff blows up when a coefficient is ~1e-17).
    scale = np.max(np.abs(ours)) or 1.0
    diff = np.max(np.abs(popt - ours)) / scale
    print(f"max coefficient difference (rel. to largest coef): {diff:.2e}")
    assert diff < 1e-5, "estimators diverged - investigate"
    print("PASS: curve_fit (Alex) and lsq_linear (ours) are equivalent")
