"""
xray_uplim.statistics
---------------------
Count-rate estimation methods shared across all observatory modules.

Three methods are implemented:

net_count_rate()
    Background-subtracted point estimate.  Not an upper limit.
    Useful as a sanity check and for comparison with the proper ULs.

kraft_upper_limit()
    Kraft, Burrows & Nousek 1991, ApJ 374, 344.
    Bayesian posterior with uniform prior on S >= 0.  Solved exactly via
    the regularised incomplete Gamma function — numerically stable for any
    observed count total, including N ~ 500+ where quad-based integration
    fails silently due to floating-point overflow.
    This is the standard method for X-ray non-detections.

gehrels_upper_limit()
    Gehrels 1986, ApJ 303, 336.
    Closed-form Poisson approximation.  Slightly overestimates at low N.
    Printed as a cross-check alongside Kraft.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm as sp_norm
from scipy.special import gammaincc


def net_count_rate(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio):
    """
    Background-subtracted net count rate.

        CR_net = (N_src - B_scaled) / t_eff

    This is a POINT ESTIMATE, not an upper limit.  A negative value is
    perfectly valid and indicates a clean non-detection (the source
    aperture collected fewer counts than the expected background).

    Uncertainty
    -----------
    Correct propagation through the area scaling:

        Var(B_scaled) = (A_src / A_bkg)^2 * N_bkg_raw

        sigma_counts  = sqrt(N_src + N_bkg_raw * area_ratio^2)
        sigma_CR      = sigma_counts / t_eff

    Note: using sqrt(N_src + B_scaled) is only correct when area_ratio = 1,
    which is never the case for a source circle + background annulus.

    Parameters
    ----------
    N_src      : int    — counts in source aperture
    B_scaled   : float  — expected background in source aperture
                          (= N_bkg_raw * area_ratio)
    t_eff      : float  — effective exposure time in seconds
    N_bkg_raw  : int    — raw background counts (before area scaling)
    area_ratio : float  — A_src / A_bkg

    Returns
    -------
    CR    : float  — net count rate in cts/s
    sigma : float  — 1-sigma Poisson uncertainty in cts/s
    """
    CR    = (N_src - B_scaled) / t_eff
    sigma = np.sqrt(N_src + N_bkg_raw * area_ratio**2) / t_eff
    return CR, sigma


def kraft_upper_limit(N_obs, B_scaled, confidence):
    """
    Kraft, Burrows & Nousek 1991 Bayesian upper limit on source signal S.

    EXACT SOLUTION via the regularised incomplete Gamma function
    ------------------------------------------------------------
    Prior      : uniform on S >= 0
    Likelihood : Poisson(N; S + B)
    Posterior  : p(S | N, B)  proportional to  (S+B)^N * exp(-(S+B))

    With the substitution lambda = S + B the normalisation integral is:

        norm  =  integral_B^inf  lambda^N * exp(-lambda)  d(lambda)
              =  Gamma(N+1, B)                [upper incomplete gamma]

    The posterior CDF therefore has the closed form:

        P(S <= s_up | N, B)
            = 1 - gammaincc(N+1, s_up+B) / gammaincc(N+1, B)

    where gammaincc(a, x) = Gamma(a, x) / Gamma(a) is scipy's regularised
    upper incomplete gamma function, evaluated in log-space internally and
    numerically stable for any N.

    Parameters
    ----------
    N_obs      : int    — counts observed in source aperture
    B_scaled   : float  — expected background in source aperture
    confidence : float  — one-sided confidence level (e.g. 0.9973)

    Returns
    -------
    S_upper : float  — upper limit on net source counts
    """
    a = float(N_obs) + 1.0
    B = float(B_scaled)

    norm_reg = gammaincc(a, B)
    if norm_reg == 0.0:
        return 0.0   # B >> N: entire posterior mass is at S ~ 0

    target = (1.0 - confidence) * norm_reg

    def equation(s_up):
        return gammaincc(a, s_up + B) - target

    # Build a bracket: s=0 gives norm_reg > target (since confidence < 1).
    # Expand s_hi until gammaincc falls below target.
    s_hi = max(float(N_obs), 1.0) + 10.0 * np.sqrt(max(float(N_obs), 1.0)) + 50.0
    for _ in range(40):
        if gammaincc(a, s_hi + B) < target:
            break
        s_hi *= 2.0

    try:
        S_upper = brentq(equation, 0.0, s_hi, xtol=1e-5, maxiter=500)
    except ValueError:
        S_upper = s_hi   # fallback; should not occur with the bracket above

    return S_upper


def gehrels_upper_limit(N_obs, B_scaled, confidence):
    """
    Gehrels 1986 closed-form Poisson upper limit.

        S_upper = (N_obs + 1 + sqrt(N_obs + 0.75) * z) - B_scaled

    where z is the one-sided Gaussian quantile for `confidence`.
    Slightly overestimates at very low N.  Useful as a cross-check.

    Parameters
    ----------
    N_obs      : int
    B_scaled   : float
    confidence : float

    Returns
    -------
    S_upper : float
    """
    z = sp_norm.ppf(confidence)
    return max(N_obs + 1.0 + np.sqrt(N_obs + 0.75) * z - B_scaled, 0.0)
