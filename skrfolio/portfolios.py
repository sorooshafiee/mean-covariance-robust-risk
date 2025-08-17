import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from .estimators import EmpiricalEstimator, LedoitWolfEstimator
from .estimators import MCDEstimator, WrappingEstimator
from .estimators import OGKEstimator, KendallTauEstimator
from .estimators import HuberMEstimator, TukeySEstimator
from .estimators import FastMCDEstimator, DetMCDEstimator
import gurobipy as gp
from gurobipy import GRB


class MarkowitzOptimizer(BaseEstimator, RegressorMixin):
    """
    Solves a regularized Markowitz portfolio optimization problem using GurobiPy.
    Regularization is L2 norm.
    """

    def __init__(
        self,
        alpha=1.0,
        rho=0.1,
        estimator_type="empirical",
        estimator_params=None,
        solver_params={"OutputFlag": 0},
    ):
        self.alpha = alpha
        self.rho = rho
        self.estimator_type = estimator_type
        self.estimator_params = estimator_params if estimator_params is not None else {}
        self.solver_params = solver_params if solver_params is not None else {}
        self.estimator_ = None
        self.weights_ = None
        self.objective_value_ = None

    def _get_internal_estimator(self):
        est_type = self.estimator_type.lower()
        if est_type == "empirical":
            return EmpiricalEstimator(**self.estimator_params)
        elif est_type == "huberm":
            return HuberMEstimator(**self.estimator_params)
        elif est_type == "ledoitwolf":
            return LedoitWolfEstimator(**self.estimator_params)
        elif est_type == "mcd":
            return MCDEstimator(**self.estimator_params)
        elif est_type == "tukeys":
            return TukeySEstimator(**self.estimator_params)
        elif est_type == "fastmcd":
            return FastMCDEstimator(**self.estimator_params)
        elif est_type == "detmcd":
            return DetMCDEstimator(**self.estimator_params)
        elif est_type == "wrapping":
            return WrappingEstimator(**self.estimator_params)
        elif est_type == "ogk":
            return OGKEstimator(**self.estimator_params)
        elif est_type == "kendalltau":
            return KendallTauEstimator(**self.estimator_params)
        else:
            raise ValueError(
                f"Unknown estimator_type: {self.estimator_type}. "
                "Ensure it's defined in _get_internal_estimator and imported."
            )

    def fit(self, X, y=None):
        """
        Fit the Markowitz model by solving the optimization problem.

        Parameters
        ----------
        X : tuple
            A tuple containing (mean_vector, covariance_matrix).
            - mean_vector : array-like of shape (n_features,)
            - covariance_matrix : array-like of shape (n_features, n_features)
        y : Ignored
            Not used, present for API consistency.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.estimator_ = self._get_internal_estimator()
        self.estimator_.fit(X)

        mu = self.estimator_.mean_
        Sigma = self.estimator_.covariance_

        m = gp.Model("Markowitz")
        if self.solver_params:
            for param_name, param_value in self.solver_params.items():
                m.setParam(param_name, param_value)

        w = m.addMVar(n_features, lb=0, ub=1, name="weights")
        s1 = m.addVar(lb=0, name="slack 1")
        s2 = m.addVar(lb=0, name="slack 2")

        m.addConstr(w @ Sigma @ w <= s1 * s1, name="risk_constraint")
        m.addConstr(w @ w <= s2 * s2, name="l2_norm_constraint")
        m.addConstr(w.sum() == 1, name="budget")

        a_rho = self.rho * np.sqrt(1 + self.alpha**2)
        objective = -mu @ w + self.alpha * s1 + a_rho * s2
        m.setObjective(objective, GRB.MINIMIZE)

        m.optimize()
        self.weights_ = np.array(w.X)
        self.objective_value_ = m.ObjVal if hasattr(m, "ObjVal") else None
        self.model = m

        return self

    def predict(self, X):
        check_is_fitted(self)
        return X @ self.weights_

    def score(self, X, y=None):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != len(self.weights_):
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but model was fit with {len(self.weights_)} features."
            )

        portfolio_returns = X @ self.weights_
        return np.mean(portfolio_returns)


class Markowitz2Optimizer(BaseEstimator, RegressorMixin):
    """
    Solves a regularized Markowitz portfolio optimization problem using GurobiPy.
    Regularization is L2 norm.
    """

    def __init__(
        self,
        alpha=1.0,
        rho=0.1,
        estimator_type="empirical",
        estimator_params=None,
        solver_params={"OutputFlag": 0},
    ):
        self.alpha = alpha
        self.rho = rho
        self.estimator_type = estimator_type
        self.estimator_params = estimator_params if estimator_params is not None else {}
        self.solver_params = solver_params if solver_params is not None else {}
        self.estimator_ = None
        self.weights_ = None
        self.objective_value_ = None

    def _get_internal_estimator(self):
        est_type = self.estimator_type.lower()
        if est_type == "empirical":
            return EmpiricalEstimator(**self.estimator_params)
        elif est_type == "huber":
            return HuberMEstimator(**self.estimator_params)
        elif est_type == "ledoitwolf":
            return LedoitWolfEstimator(**self.estimator_params)
        elif est_type == "mcd":
            return MCDEstimator(**self.estimator_params)
        elif est_type == "wrapping":
            return WrappingEstimator(**self.estimator_params)
        elif est_type == "ogk":
            return OGKEstimator(**self.estimator_params)
        elif est_type == "kendalltau":
            return KendallTauEstimator(**self.estimator_params)
        else:
            raise ValueError(
                f"Unknown estimator_type: {self.estimator_type}. "
                "Ensure it's defined in _get_internal_estimator and imported."
            )

    def fit(self, X, y=None):
        """
        Fit the Markowitz2 model by solving the optimization problem.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Asset returns matrix.
        y : Ignored
            Not used, present for API consistency.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.estimator_ = self._get_internal_estimator()
        self.estimator_.fit(X)

        mu = self.estimator_.mean_
        Sigma = self.estimator_.covariance_

        m = gp.Model("Markowitz2")
        if self.solver_params:
            for param_name, param_value in self.solver_params.items():
                m.setParam(param_name, param_value)

        w = m.addMVar(n_features, lb=0, ub=1, name="weights")

        m.addConstr(w.sum() == 1, name="budget")

        objective = -mu @ w + self.alpha * w @ Sigma @ w + self.rho * w @ w
        m.setObjective(objective, GRB.MINIMIZE)

        m.optimize()
        self.weights_ = np.array(w.X)
        self.objective_value_ = m.ObjVal if hasattr(m, "ObjVal") else None
        self.model = m

        return self

    def predict(self, X):
        check_is_fitted(self)
        return X @ self.weights_

    def score(self, X, y=None):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != len(self.weights_):
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but model was fit with {len(self.weights_)} features."
            )

        portfolio_returns = X @ self.weights_
        return np.mean(portfolio_returns)


class WassersteinOptimizer(BaseEstimator, RegressorMixin):
    """
    Solves a Wasserstein portfolio optimization problem using GurobiPy.
    """

    def __init__(self, beta=1.0, rho=0.1, solver_params={"OutputFlag": 0}):
        self.beta = beta
        self.rho = rho
        self.solver_params = solver_params if solver_params is not None else {}
        self.weights_ = None
        self.objective_value_ = None

    def fit(self, X, y=None):
        """
        Fit the Wasserstein model by solving the optimization problem.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Asset returns matrix.
        y : Ignored
            Not used, present for API consistency.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        m = gp.Model("Wasserstein")
        if self.solver_params:
            for param_name, param_value in self.solver_params.items():
                m.setParam(param_name, param_value)

        w = m.addMVar(n_features, lb=0, ub=1, name="weights")
        w0 = m.addVar(lb=0, name="slack_0")
        tau = m.addVar(lb=-GRB.INFINITY, name="tau")
        s = m.addMVar(n_samples, lb=0, name="slack")

        m.addConstr(w.sum() == 1, name="budget")
        m.addConstr(w @ w <= w0 * w0, name="l2_norm_constraint")
        m.addConstr(s >= -X @ w - tau, name="cvar_constraint")

        if self.beta == 0:
            raise ValueError("beta must be non-zero")
        objective = self.beta * tau + self.rho * w0 + s.sum() / n_samples
        m.setObjective(objective, GRB.MINIMIZE)

        m.optimize()
        self.weights_ = np.array(w.X)
        self.objective_value_ = m.ObjVal / self.beta if hasattr(m, "ObjVal") else None
        self.model = m

        return self

    def predict(self, X):
        check_is_fitted(self)
        return X @ self.weights_

    def score(self, X, y=None):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != len(self.weights_):
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but model was fit with {len(self.weights_)} features."
            )

        portfolio_returns = X @ self.weights_
        return np.mean(portfolio_returns)


class ChiSquaredOptimizer(BaseEstimator, RegressorMixin):
    """
    Solves a Chi-squared portfolio optimization problem using GurobiPy.
    """

    def __init__(self, beta=1.0, rho=0.1, solver_params={"OutputFlag": 0}):
        self.beta = beta
        self.rho = rho
        self.solver_params = solver_params if solver_params is not None else {}
        self.weights_ = None
        self.objective_value_ = None

    def fit(self, X, y=None):
        """
        Fit the Chi-squared model by solving the optimization problem.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Asset returns matrix.
        y : Ignored
            Not used, present for API consistency.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        m = gp.Model("ChiSquared")
        if self.solver_params:
            for param_name, param_value in self.solver_params.items():
                m.setParam(param_name, param_value)

        w = m.addMVar(n_features, lb=0, ub=1, name="weights")
        s = m.addMVar(n_samples, lb=0, name="slack")
        s0 = m.addVar(lb=0, name="slack_0")
        tau = m.addVar(lb=-GRB.INFINITY, name="tau")
        eta = m.addVar(lb=-GRB.INFINITY, name="eta")

        m.addConstr(w.sum() == 1, name="budget")
        m.addConstr(s @ s / n_samples <= s0 * s0, name="l2_norm_constraint")
        m.addConstr(s >= -X @ w - tau - eta, name="cvar_constraint")
        m.addConstr(s >= -eta, name="eta_constraint")

        if self.beta == 0:
            raise ValueError("beta must be non-zero")
        rho2 = (1 + 2 * self.rho) ** 0.5
        objective = self.beta * tau + rho2 * s0 + eta
        m.setObjective(objective, GRB.MINIMIZE)

        m.optimize()
        self.weights_ = np.array(w.X)
        self.objective_value_ = m.ObjVal if hasattr(m, "ObjVal") else None
        self.model = m

        return self

    def predict(self, X):
        check_is_fitted(self)
        return X @ self.weights_

    def score(self, X, y=None):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != len(self.weights_):
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but model was fit with {len(self.weights_)} features."
            )

        portfolio_returns = X @ self.weights_
        return np.mean(portfolio_returns)


class SAAOptimizer(BaseEstimator, RegressorMixin):
    """
    Solves a SAA portfolio optimization problem using GurobiPy.
    """

    def __init__(self, beta=1.0, solver_params={"OutputFlag": 0}):
        self.beta = beta
        self.solver_params = solver_params if solver_params is not None else {}
        self.weights_ = None
        self.objective_value_ = None

    def fit(self, X, y=None):
        """
        Fit the SAA model by solving the optimization problem.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Asset returns matrix.
        y : Ignored
            Not used, present for API consistency.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        m = gp.Model("SAA")
        if self.solver_params:
            for param_name, param_value in self.solver_params.items():
                m.setParam(param_name, param_value)

        w = m.addMVar(n_features, lb=0, ub=1, name="weights")
        tau = m.addVar(lb=-GRB.INFINITY, name="tau")
        s = m.addMVar(n_samples, lb=0, name="slack")

        m.addConstr(w.sum() == 1, name="budget")
        m.addConstr(s >= -X @ w - tau, name="cvar_constraint")

        if self.beta == 0:
            raise ValueError("beta must be non-zero")
        objective = self.beta * tau + s.sum() / n_samples
        m.setObjective(objective, GRB.MINIMIZE)

        m.optimize()
        self.weights_ = np.array(w.X)
        self.objective_value_ = m.ObjVal / self.beta if hasattr(m, "ObjVal") else None
        self.model = m

        return self

    def predict(self, X):
        check_is_fitted(self)
        return X @ self.weights_

    def score(self, X, y=None):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != len(self.weights_):
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but model was fit with {len(self.weights_)} features."
            )

        portfolio_returns = X @ self.weights_
        return np.mean(portfolio_returns)
