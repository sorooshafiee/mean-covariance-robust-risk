import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2
from scipy.linalg import solve
from sklearn.covariance import empirical_covariance


def _initial_estimates(X, initial_location, initial_covariance, min_cov_diag=1e-8):
    n_features = X.shape[1]
    if isinstance(initial_location, str):
        if initial_location.lower() == "empirical":
            loc = np.mean(X, axis=0)
        elif initial_location.lower() == "median":
            loc = np.median(X, axis=0)
        else:
            raise ValueError("Invalid string for initial_location.")
    elif isinstance(initial_location, np.ndarray):
        loc = np.asarray(initial_location).flatten()
        if loc.shape[0] != n_features:
            raise ValueError("Provided initial_location has wrong dimensions.")
    else:
        raise TypeError(
            "initial_location must be 'empirical', 'median', or np.ndarray."
        )

    if isinstance(initial_covariance, str):
        if initial_covariance.lower() == "empirical":
            cov = empirical_covariance(X, assume_centered=False)
        elif initial_covariance.lower() == "mad":
            mad_scales = 1.4826 * np.median(np.abs(X - np.median(X, axis=0)), axis=0)
            mad_scales[mad_scales < min_cov_diag] = min_cov_diag
            cov = np.diag(mad_scales**2)
        else:
            raise ValueError("Invalid string for initial_covariance.")
    elif isinstance(initial_covariance, np.ndarray):
        cov = np.asarray(initial_covariance)
        if cov.shape != (n_features, n_features):
            raise ValueError("Provided initial_covariance has wrong dimensions.")
    else:
        raise TypeError("initial_covariance must be 'empirical', 'mad', or np.ndarray.")

    return loc, cov


def _compute_mahalanobis_distances(X, mu, Sigma):
    diff_X_mu = X - mu
    Z = solve(Sigma, diff_X_mu.T)
    d2 = np.sum(diff_X_mu * Z.T, axis=1)
    return np.maximum(np.sqrt(d2), 1e-8)


def _huber_w(u, eps):
    u_abs = np.abs(u)
    val = np.where(u_abs <= eps, 1.0, eps / u_abs)
    return val


def _calculate_huber_eps(n_features, q=0.95):
    return np.sqrt(chi2.ppf(q, df=n_features))


def _tukey_rho(u, eps):
    u_abs = np.abs(u)
    val = np.where(
        u_abs <= eps, (eps**2 / 6.0) * (1 - (1 - (u_abs / eps) ** 2) ** 3), eps**2 / 6.0
    )
    return val


def _tukey_psi(u, eps):
    u_abs = np.abs(u)
    val = np.where(u_abs <= eps, u * (1 - (u_abs / eps) ** 2) ** 2, 0.0)
    return val


def _tukey_w(u, eps):
    u_abs = np.abs(u)
    val = np.where(u_abs <= eps, (1 - (u_abs / eps) ** 2) ** 2, 0.0)
    return val


def _calculate_tukey_eps(n_features, c=4.685):
    # Based on Wilson-Hilferty transformation
    coef = 2 / (9 * n_features)
    term1 = np.sqrt(coef * c)
    term2 = 1 - coef
    main_term = term1 + term2
    eps_2 = n_features * (main_term**3)
    eps = np.sqrt(eps_2)
    return eps


def _calculate_tukey_b0(n_features, eps):
    v = n_features
    eps_2 = eps**2
    eps_4 = eps**4
    term1 = v * chi2.cdf(eps_2, df=v + 2) / 2.0

    den2 = 2.0 * eps_2
    term2 = v * (v + 2) * chi2.cdf(eps_2, df=v + 4) / den2

    den3 = 6.0 * eps_4
    term3 = v * (v + 2) * (v + 4) * chi2.cdf(eps_2, df=v + 6) / den3

    term4 = (eps_2 / 6.0) * (1.0 - chi2.cdf(eps_2, df=v))

    b0 = term1 - term2 + term3 + term4

    return b0


def _solve_for_tukey_scale(d_values, b0, eps, initial_s_guess):
    """
    Solves (1/N) * sum(rho(d_i/s, eps)) = b0 for s
    """
    objective_func = lambda s: (np.mean(_tukey_rho(d_values / s, eps)) - b0)
    a = initial_s_guess / 2
    b = initial_s_guess * 2
    val_a = objective_func(a)
    val_b = objective_func(b)
    while val_a * val_b >= 0:
        if val_a > 0:
            a /= 2
            val_a = objective_func(a)
            b = a
            val_b = objective_func(b)
        else:
            b *= 2
            val_b = objective_func(b)
            a = b
            val_a = objective_func(a)
    return brentq(objective_func, a, b)
