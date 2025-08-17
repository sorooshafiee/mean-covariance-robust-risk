import numpy as np
from .utils import _initial_estimates, _compute_mahalanobis_distances
from .utils import _huber_w, _calculate_huber_eps, _tukey_rho
from .utils import _tukey_psi, _tukey_w, _calculate_tukey_eps
from .utils import _calculate_tukey_b0, _solve_for_tukey_scale
from scipy.stats import chi, chi2
from scipy.optimize import brentq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
from sklearn.covariance import empirical_covariance
from sklearn.utils.validation import check_array, check_is_fitted
from robpy.covariance import FastMCD, WrappingCovariance, OGK, KendallTau, DetMCD


class EmpiricalEstimator(BaseEstimator, TransformerMixin):
    """
    Estimates mean and covariance using empirical (sample) calculations.
    """

    def __init__(self, **estimator_params):
        self.estimator_params = estimator_params

    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        estimator = EmpiricalCovariance(**self.estimator_params)
        estimator.fit(X)
        self.mean_ = estimator.location_
        self.covariance_ = estimator.covariance_
        self.estimator_ = estimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.mean_, self.covariance_


class LedoitWolfEstimator(BaseEstimator, TransformerMixin):
    """
    Estimates mean (empirical) and covariance using Ledoit-Wolf shrinkage.
    """

    def __init__(self, **estimator_params):
        self.estimator_params = estimator_params

    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        estimator = LedoitWolf(**self.estimator_params)
        estimator.fit(X)
        self.mean_ = estimator.location_
        self.covariance_ = estimator.covariance_
        self.estimator_ = estimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.mean_, self.covariance_


class MCDEstimator(BaseEstimator, TransformerMixin):
    """
    MCD Estimator based on scikit-learn library.
    """

    def __init__(self, **estimator_params):
        self.estimator_params = estimator_params

    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        if X.shape[0] < X.shape[1]:
            estimator = EmpiricalCovariance(**self.estimator_params)
            estimator.fit(X)
            self.mean_ = estimator.location_
            self.covariance_ = estimator.covariance_
            self.estimator_ = estimator
        else:
            estimator = MinCovDet(**self.estimator_params)
            estimator.fit(X)
            self.mean_ = estimator.location_
            self.covariance_ = estimator.covariance_
            self.estimator_ = estimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.mean_, self.covariance_


class DetMCDEstimator(BaseEstimator, TransformerMixin):
    """
    Estimates mean and covariance using the DetMCD
    from the RobPy library.
    """

    def __init__(self, **estimator_params):
        self.estimator_params = estimator_params

    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        estimator = DetMCD(**self.estimator_params)
        estimator.fit(X)
        self.mean_ = estimator.location_
        self.covariance_ = estimator.covariance_
        self.estimator_ = estimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.mean_, self.covariance_


class FastMCDEstimator(BaseEstimator, TransformerMixin):
    """
    Estimates mean and covariance using the FastMCD
    from the RobPy library.
    """

    def __init__(self, **estimator_params):
        self.estimator_params = estimator_params

    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        estimator = FastMCD(**self.estimator_params)
        estimator.fit(X)
        self.mean_ = estimator.location_
        self.covariance_ = estimator.covariance_
        self.estimator_ = estimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.mean_, self.covariance_


class WrappingEstimator(BaseEstimator, TransformerMixin):
    """
    Estimates mean and covariance using a robust wrapping estimator
    from the RobPy library.
    """

    def __init__(self, **estimator_params):
        self.estimator_params = estimator_params

    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        estimator = WrappingCovariance(**self.estimator_params)
        estimator.fit(X)
        self.mean_ = estimator.location_
        self.covariance_ = estimator.covariance_
        self.estimator_ = estimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.mean_, self.covariance_


class OGKEstimator(BaseEstimator, TransformerMixin):
    """
    Estimates mean and covariance using the Orthogonalized Gnanadesikan-Kettenring (OGK)
    estimator from the RobPy library.
    """

    def __init__(self, **estimator_params):
        self.estimator_params = estimator_params

    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        estimator = OGK(**self.estimator_params)
        estimator.fit(X)
        self.mean_ = estimator.location_
        self.covariance_ = estimator.covariance_
        self.estimator_ = estimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.mean_, self.covariance_


class KendallTauEstimator(BaseEstimator, TransformerMixin):
    """
    Estimates mean and covariance using the Kendall's Tau
    estimator from the RobPy library.
    """

    def __init__(self, **estimator_params):
        self.estimator_params = estimator_params

    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        estimator = KendallTau(**self.estimator_params)
        estimator.fit(X)
        self.mean_ = estimator.location_
        self.covariance_ = estimator.covariance_
        self.estimator_ = estimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.mean_, self.covariance_


class HuberMEstimator(BaseEstimator, TransformerMixin):
    """
    Huber's M-Estimator for mean and covariance using Huber's weight functions.

    This implementation follows Huber (1986), using Huber-type weights.

    The iterative equations are:
    d_i^2 = (x_i - mu)^T Sigma^-1 (x_i - mu)
    w_i = min(1, k / sqrt(d_i^2))
    mu_new = sum(w_i * x_i) / sum(w_i)
    Sigma_new = (1/n_samples) * sum(w_i * (x_i - mu_new)(x_i - mu_new)^T)

    Parameters
    ----------
    q : float, default=0.95
        Quantile for Huber's weight function. This is used to determine the
        threshold for the weight function, corresponding to the 95% quantile
        of the chi-squared distribution with `n_features` degrees of freedom.

    eps : float, default=None
        Tuning constant for Huber's weight function. Corresponds to 95%
        efficiency for the univariate normal distribution.
        Commonly, for m dimensions, eps is chosen as sqrt(chi2.ppf(0.95, df=m))
        or similar, but here we use a fixed eps as in univariate case for simplicity,
        or user can set it.

    max_iter : int, default=1000
        Maximum number of iterations for the IRLS algorithm.

    tol : float, default=1e-6
        Tolerance for convergence. Iterations stop when the change in
        location and scatter estimates is below this threshold.

    initial_location : str or np.ndarray, default='empirical'
        Method to compute initial location:
        - 'empirical': empirical mean.
        - 'median': component-wise median.
        - np.ndarray: user-provided initial location vector.

    initial_covariance : str or np.ndarray, default='empirical'
        Method to compute initial covariance:
        - 'empirical': empirical covariance.
        - 'mad': diagonal matrix with squared Median Absolute Deviations (scaled).
        - np.ndarray: user-provided initial covariance matrix.
        Ignored if initial_location is an ndarray (then this must also be one).

    min_cov_diag : float, default=1e-8
        Small constant added to the diagonal of the covariance matrix during
        iterations if it's found to be non-positive definite or to prevent
        singularity in initial MAD estimates if a feature is constant.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Estimated robust location (mean).

    covariance_ : np.ndarray of shape (n_features, n_features)
        Estimated robust scatter (covariance) matrix.

    n_iter_ : int
        Number of iterations run.

    converged_ : bool
        True if the algorithm converged.
    """

    def __init__(
        self,
        q=0.95,
        eps=None,
        max_iter=1000,
        tol=1e-6,
        initial_location="empirical",
        initial_covariance="empirical",
        min_cov_diag=1e-8,
    ):
        self.q = q
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        self.initial_location = initial_location
        self.initial_covariance = initial_covariance
        self.min_cov_diag = min_cov_diag

    def fit(self, X, y=None):
        """
        Fit the Huber M-estimator to the data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        n_samples, n_features = X.shape

        mu_curr, Sigma_curr = _initial_estimates(
            X, self.initial_location, self.initial_covariance, self.min_cov_diag
        )
        if self.eps is None:
            self.eps = _calculate_huber_eps(n_features, self.q)
        self.converged_ = False

        for i in range(self.max_iter):
            mu_prev = mu_curr.copy()
            Sigma_prev = Sigma_curr.copy()

            Sigma_curr_reg = Sigma_curr + np.eye(n_features) * self.min_cov_diag
            d = _compute_mahalanobis_distances(X, mu_curr, Sigma_curr_reg)

            weights = _huber_w(d, self.eps)
            sum_weights = np.sum(weights)

            mu_curr = np.sum(weights[:, np.newaxis] * X, axis=0) / sum_weights

            diff_from_new_mu = X - mu_curr
            Sigma_curr = (
                diff_from_new_mu.T @ (weights[:, np.newaxis] * diff_from_new_mu)
            ) / n_samples

            loc_change = np.linalg.norm(mu_curr - mu_prev) / (
                np.linalg.norm(mu_prev) + 1e-8
            )
            cov_change = np.linalg.norm(Sigma_curr - Sigma_prev, "fro") / (
                np.linalg.norm(Sigma_prev, "fro") + 1e-8
            )

            if loc_change < self.tol and cov_change < self.tol:
                self.converged_ = True
                break

        self.n_iter_ = i + 1
        if not self.converged_ and self.n_iter_ == self.max_iter:
            print(
                f"Warning: M-estimator did not converge within {self.max_iter} iterations. "
            )

        self.mean_ = mu_curr
        self.covariance_ = Sigma_curr
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.mean_, self.covariance_


class TukeySEstimator(BaseEstimator, TransformerMixin):
    """
    S-Estimator for location and scatter using Tukey's biweight functions,
    following the algorithm structure from Campbell, Lopuhaä, and Rousseeuw (1998).

    Parameters
    ----------
    c : float, default=4.685
        Tuning constant for Tukey's biweight function.
        This constant balances robustness and efficiency.

    eps : float, default=None
        Thresholding constant for Tukey's weight function.
        If None, it is calculated based on the number of features.

    max_iter : int, default=1000
        Maximum number of iterations for the algorithm.

    tol : float, default=1e-6
        Tolerance for convergence. Iterations stop when the relative change in
        location and scatter estimates is below this threshold.

    initial_location : str or np.ndarray, default='empirical'
        Method to compute initial location for S2 step.
        Options: 'empirical' (mean), 'median'.

    initial_covariance : str or np.ndarray, default='empirical'
        Method to compute initial covariance for S2 step.
        Options: 'empirical', 'mad' (scaled MAD for diagonal).

    min_cov_diag : float, default=1e-8
        Small constant added to the diagonal of covariance matrices before inversion
        to improve numerical stability.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Estimated robust location (mean).

    covariance_ : np.ndarray of shape (n_features, n_features)
        Estimated robust scatter (covariance) matrix.

    scale_ : float
        The final scale factor `s` such that covariance_ = s^2 * shape_matrix.

    b0_ : float
        The consistency constant E[rho(D)] used in the constraint.

    n_iter_ : int
        Number of iterations run.

    converged_ : bool
        True if the algorithm converged.
    """

    def __init__(
        self,
        c=4.685,
        eps=None,
        max_iter=1000,
        tol=1e-6,
        initial_location="empirical",
        initial_covariance="empirical",
        min_cov_diag=1e-8,
    ):
        self.c = c
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        self.initial_location = initial_location
        self.initial_covariance = initial_covariance
        self.min_cov_diag = min_cov_diag

    def fit(self, X, y=None):
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        n_samples, n_features = X.shape
        if self.eps is None:
            self.eps = _calculate_tukey_eps(n_features, self.c)

        self.b0_ = _calculate_tukey_b0(n_features, self.eps)

        # Initial estimates
        mu_curr, Sigma_unscaled_curr = _initial_estimates(
            X, self.initial_location, self.initial_covariance, self.min_cov_diag
        )

        # Scale initial covariance to satisfy constraint
        Sigma_unscaled_curr_reg = (
            Sigma_unscaled_curr + np.eye(n_features) * self.min_cov_diag
        )

        d = _compute_mahalanobis_distances(X, mu_curr, Sigma_unscaled_curr_reg)

        s0_median_chi_val = chi.ppf(0.5, df=n_features)
        s0_initial_guess = np.median(d) / s0_median_chi_val
        if s0_initial_guess < 1e-4 or not np.isfinite(s0_initial_guess):
            s0_initial_guess = 1.0

        s0 = _solve_for_tukey_scale(d, self.b0_, self.eps, s0_initial_guess)
        if not np.isfinite(s0) or s0 < 1e-8:
            s0 = 1.0

        Sigma_curr = s0**2 * Sigma_unscaled_curr
        s_curr = s0

        # Update the estimates iteratively
        self.converged_ = False
        for i in range(self.max_iter):
            mu_prev = mu_curr.copy()
            Sigma_prev = Sigma_curr.copy()

            Sigma_curr_reg = Sigma_curr + np.eye(n_features) * self.min_cov_diag
            d = _compute_mahalanobis_distances(X, mu_curr, Sigma_curr_reg)

            weights = _tukey_w(d, self.eps)
            sum_weights = np.sum(weights)

            mu_curr = np.sum(weights[:, np.newaxis] * X, axis=0) / sum_weights

            diff_X_new_mu = X - mu_curr
            V_shape_next = (
                diff_X_new_mu.T @ (weights[:, np.newaxis] * diff_X_new_mu)
            ) / sum_weights

            V_shape_next_reg = V_shape_next + np.eye(n_features) * self.min_cov_diag
            d = _compute_mahalanobis_distances(X, mu_curr, V_shape_next_reg)

            s_next = _solve_for_tukey_scale(d, self.b0_, self.eps, s_curr)
            Sigma_curr = s_next**2 * V_shape_next
            s_curr = s_next

            loc_change = np.linalg.norm(mu_curr - mu_prev) / (
                np.linalg.norm(mu_prev) + 1e-8
            )
            cov_change = np.linalg.norm(Sigma_curr - Sigma_prev, "fro") / (
                np.linalg.norm(Sigma_prev, "fro") + 1e-8
            )

            if loc_change < self.tol and cov_change < self.tol:
                self.converged_ = True
                break

        self.n_iter_ = i + 1
        if not self.converged_ and self.n_iter_ == self.max_iter:
            print(
                f"Warning: S-estimator did not converge within {self.max_iter} iterations. "
                f"Loc change: {loc_change:.2e}, Cov change: {cov_change:.2e}"
            )
        elif not self.converged_:
            print(
                f"S-estimator stopped at iteration {self.n_iter_} due to an issue (e.g. singular matrix, zero weights, or scale collapse)."
            )

        self.mean_ = mu_curr
        self.covariance_ = Sigma_curr
        self.scale_ = s_curr
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.mean_, self.covariance_
