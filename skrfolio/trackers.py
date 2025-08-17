import os
import numpy as np
from scipy.linalg import sqrtm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from .estimators import EmpiricalEstimator, LedoitWolfEstimator
from .estimators import MCDEstimator, WrappingEstimator
from .estimators import OGKEstimator, KendallTauEstimator
from .estimators import HuberMEstimator, TukeySEstimator
from .estimators import FastMCDEstimator, DetMCDEstimator
from juliacall import Main as jl
import gurobipy as gp
from gurobipy import GRB

_CURRENT_DIR = os.path.dirname(__file__)
_GELBRICH_FILE_PATH = os.path.join(_CURRENT_DIR, "gelbrich.jl")
_DELAGE_JULIA_FILE = os.path.join(_CURRENT_DIR, "delage.jl")
_CHEBYSHEV_JULIA_FILE = os.path.join(_CURRENT_DIR, "chebyshev.jl")

try:
    jl.seval(f'include("{_GELBRICH_FILE_PATH}")')
    jl.seval(f'include("{_DELAGE_JULIA_FILE}")')
    jl.seval(f'include("{_CHEBYSHEV_JULIA_FILE}")')
    print("\n Julia solver files loaded successfully.")
except Exception as e:
    print(f"Failed to load Julia components: {e}")


class SAAOptimizer(BaseEstimator, RegressorMixin):
    """
    Solves the SAA problem using GurobiPy.

    Parameters
    ----------
    p : {1, 2}, default=2
        The norm used to measure the tracking error.
        - p=1 corresponds to the L1-norm (sum of absolute errors).
        - p=2 corresponds to the L2-norm (mean square error).

    solver_params : dict, optional
        Parameters to be passed to the Gurobi solver. For example,
        to suppress solver output, you can pass `{'OutputFlag': 0}`.
    """

    def __init__(
        self,
        p=2,
        solver_params={"OutputFlag": 0},
    ):
        self.p = p
        self.solver_params = solver_params if solver_params is not None else {}
        self.weights_ = None
        self.objective_value_ = None
        self.model_ = None
        self.termination_status_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Fit the Wasserstein model by solving the optimization problem.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Asset returns matrix, where the last column is the index return.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        n_assets = n_features - 1

        m = gp.Model("SAAOptimizer")
        if self.solver_params:
            for param, value in self.solver_params.items():
                m.setParam(param, value)

        w = m.addMVar(n_features, lb=-GRB.INFINITY)
        err_vector = m.addMVar(n_samples, lb=-GRB.INFINITY)
        err = m.addVar()

        m.addConstr(w[n_assets] == -1)
        m.addConstr(w[:n_assets].sum() == 1)
        m.addConstr(w[:n_assets] >= 0)
        m.addConstr(err_vector == X @ w)
        m.addGenConstrNorm(err, err_vector, float(self.p))
        if self.p == 1:
            objective = (1 / n_samples) * err
        elif self.p == 2:
            objective = (1 / np.sqrt(n_samples)) * err
        else:
            raise ValueError("p must be 1 or 2.")

        m.setObjective(objective, GRB.MINIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            self.weights_ = w.X
            self.objective_value_ = m.ObjVal
            self.termination_status_ = "OPTIMAL"
        else:
            self.weights_ = None
            self.objective_value_ = None
            self.termination_status_ = f"NON-OPTIMAL (Status: {m.Status})"

        self.model_ = m
        return self

    def predict(self, X):
        """
        Calculates the portfolio returns for a given set of asset returns.
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fit with {self.n_features_in_} features."
            )
        return X @ self.weights_

    def score(self, X, y=None):
        """
        Calculates the performance score for GridSearchCV.

        The convention in scikit-learn is that a higher score is better.
        Since we want to minimize tracking error, the score is defined
        as the NEGATIVE mean tracking error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The validation data, containing asset and index returns.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        score : float
            The negative mean tracking error.
        """
        check_is_fitted(self)

        # Calculate the tracking error on the validation set X
        asset_returns = X[:, :-1]
        index_returns = X[:, -1]
        asset_weights = self.weights_[:-1]
        portfolio_returns = asset_returns @ asset_weights

        if self.p == 1:
            tracking_errors = np.abs(portfolio_returns - index_returns)
        elif self.p == 2:
            tracking_errors = (portfolio_returns - index_returns) ** 2
        else:
            raise ValueError("p must be 1 or 2.")

        mean_tracking_error = np.mean(tracking_errors)

        return -mean_tracking_error


class ChiSquaredOptimizer(BaseEstimator, RegressorMixin):
    """
    Solves a regularized Chi-squared problem using GurobiPy.

    Parameters
    ----------
    rho : float, default=0.1
        The radius of the Chi-squared ambiguity set.

    p : {1, 2}, default=2
        The norm used to measure the tracking error in the penalty term.

    solver_params : dict, optional
        Parameters to be passed to the Gurobi solver. For example,
        to suppress solver output, you can pass `{'OutputFlag': 0}`.
    """

    def __init__(
        self,
        rho=0.1,
        p=2,
        solver_params={"OutputFlag": 0},
    ):
        self.rho = rho
        self.p = p
        self.solver_params = solver_params if solver_params is not None else {}
        self.weights_ = None
        self.objective_value_ = None
        self.model_ = None
        self.termination_status_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Fit the Chi-squared model by solving the optimization problem.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Asset returns matrix, where the last column is the index return.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        n_assets = n_features - 1

        if self.rho <= 1e-6:
            optimizer = SAAOptimizer(p=self.p, solver_params=self.solver_params).fit(X)
            self.weights_ = optimizer.weights_
            self.objective_value_ = optimizer.objective_value_
            self.termination_status_ = optimizer.termination_status_
            self.model_ = optimizer.model_
            return self

        c = np.sqrt(1 + 2 * self.rho)
        m = gp.Model("ChiSquaredOptimizer")
        if self.solver_params:
            for param, value in self.solver_params.items():
                m.setParam(param, value)

        w = m.addMVar(n_features, lb=-GRB.INFINITY)
        eta = m.addVar(lb=-GRB.INFINITY)
        s = m.addMVar(n_samples, lb=0)

        m.addConstr(w[n_assets] == -1)
        m.addConstr(w[:n_assets].sum() == 1)
        m.addConstr(w[:n_assets] >= 0)
        d = m.addMVar(n_samples, lb=-GRB.INFINITY)
        m.addConstr(d == X @ w)

        if self.p == 1:
            abs_d = m.addMVar(n_samples, lb=0, name="abs_error")
            for i in range(n_samples):
                m.addGenConstrAbs(abs_d[i], d[i])
            m.addConstr(abs_d - eta <= s)
        elif self.p == 2:
            for i in range(n_samples):
                m.addConstr(d[i] * d[i] - eta <= s[i])
        else:
            raise ValueError("p must be 1 or 2.")

        objective = eta + (c / np.sqrt(n_samples)) * s.sum()
        m.setObjective(objective, GRB.MINIMIZE)

        m.optimize()

        if m.Status == GRB.OPTIMAL:
            self.weights_ = w.X
            self.objective_value_ = m.ObjVal
            self.termination_status_ = "OPTIMAL"
        else:
            self.weights_ = None
            self.objective_value_ = None
            self.termination_status_ = f"NON-OPTIMAL (Status: {m.Status})"

        self.model_ = m
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fit with {self.n_features_in_} features."
            )
        return X @ self.weights_

    def score(self, X, y=None):
        check_is_fitted(self)
        asset_returns = X[:, :-1]
        index_returns = X[:, -1]
        asset_weights = self.weights_[:-1]
        portfolio_returns = asset_returns @ asset_weights

        if self.p == 1:
            tracking_errors = np.abs(portfolio_returns - index_returns)
        elif self.p == 2:
            tracking_errors = (portfolio_returns - index_returns) ** 2
        else:
            raise ValueError("p must be 1 or 2.")
        mean_tracking_error = np.mean(tracking_errors)

        return -mean_tracking_error


class WassersteinOptimizer(BaseEstimator, RegressorMixin):
    """
    Solves a regularized Wasserstein problem using GurobiPy.

    Parameters
    ----------
    rho : float, default=0.1
        The regularization parameter that controls the penalty on the L2 norm
        of the portfolio weights.

    p : {1, 2}, default=2
        The norm used to measure the tracking error.
        - p=1 corresponds to the L1-norm (sum of absolute errors).
        - p=2 corresponds to the L2-norm (mean square error).

    solver_params : dict, optional
        Parameters to be passed to the Gurobi solver. For example,
        to suppress solver output, you can pass `{'OutputFlag': 0}`.
    """

    def __init__(
        self,
        rho=0.1,
        p=2,
        solver_params={"OutputFlag": 0},
    ):
        self.rho = rho
        self.p = p
        self.solver_params = solver_params if solver_params is not None else {}
        self.weights_ = None
        self.objective_value_ = None
        self.model_ = None
        self.termination_status_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Fit the Wasserstein model by solving the optimization problem.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Asset returns matrix, where the last column is the index return.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        n_assets = n_features - 1

        if self.rho <= 1e-6:
            optimizer = SAAOptimizer(p=self.p, solver_params=self.solver_params).fit(X)
            self.weights_ = optimizer.weights_
            self.objective_value_ = optimizer.objective_value_
            self.termination_status_ = optimizer.termination_status_
            self.model_ = optimizer.model_
            return self

        m = gp.Model("WassersteinOptimizer")
        if self.solver_params:
            for param, value in self.solver_params.items():
                m.setParam(param, value)

        w = m.addMVar(n_features, lb=-GRB.INFINITY)
        error_vector = m.addMVar(n_samples, lb=-GRB.INFINITY)
        error = m.addVar()
        reg_norm = m.addVar()

        m.addConstr(w[n_assets] == -1)
        m.addConstr(gp.quicksum(w[i] for i in range(n_assets)) == 1)
        m.addConstr(w[:n_assets] >= 0)
        m.addConstr(error_vector == X @ w)
        m.addGenConstrNorm(error, error_vector, float(self.p))
        m.addGenConstrNorm(reg_norm, w, 2)
        if self.p == 1:
            objective = (1 / n_samples) * error + self.rho * reg_norm
        elif self.p == 2:
            objective = (1 / np.sqrt(n_samples)) * error + self.rho * reg_norm
        else:
            raise ValueError("p must be 1 or 2.")

        m.setObjective(objective, GRB.MINIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            self.weights_ = w.X
            self.objective_value_ = m.ObjVal
            self.termination_status_ = "OPTIMAL"
        else:
            self.weights_ = None
            self.objective_value_ = None
            self.termination_status_ = f"NON-OPTIMAL (Status: {m.Status})"

        self.model_ = m
        return self

    def predict(self, X):
        """
        Calculates the portfolio returns for a given set of asset returns.
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fit with {self.n_features_in_} features."
            )
        return X @ self.weights_

    def score(self, X, y=None):
        check_is_fitted(self)
        asset_returns = X[:, :-1]
        index_returns = X[:, -1]
        asset_weights = self.weights_[:-1]
        portfolio_returns = asset_returns @ asset_weights

        if self.p == 1:
            tracking_errors = np.abs(portfolio_returns - index_returns)
        elif self.p == 2:
            tracking_errors = (portfolio_returns - index_returns) ** 2
        else:
            raise ValueError("p must be 1 or 2.")
        mean_tracking_error = np.mean(tracking_errors)

        return -mean_tracking_error


class DelageOptimizer(BaseEstimator, RegressorMixin):
    """
    Solves a DRO problem based on the work of Delage & Ye, using JuMP and MOSEK.

    Parameters
    ----------
    rho : float, default=0.1
        The radius of the ambiguity set. Determines the level of robustness.

    k : float, default=0.5
        A parameter controlling the shape of the ambiguity set.

    p : {1, 2}, default=2
        The norm used for the loss function in the DRO formulation.

    estimator_type : str, default='empirical'
        The type of mean/covariance estimator to use.

    solver_params : dict, optional
        Parameters to be passed to the JuMP/MOSEK solver.
    """

    def __init__(
        self,
        rho=0.1,
        k=0.5,
        p=2,
        estimator_type="empirical",
        estimator_params=None,
        solver_params={"silent": True},
    ):
        self.rho = rho
        self.k = k
        self.p = p
        self.estimator_type = estimator_type
        self.estimator_params = estimator_params if estimator_params is not None else {}
        self.solver_params = solver_params if solver_params is not None else {}
        self.weights_ = None
        self.objective_value_ = None
        self.termination_status_ = None
        self.n_features_in_ = None

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
        Fit the Delage model by calling the external Julia solver.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Asset returns matrix, where the last column is the index return.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        self.n_features_in_ = X.shape[1]

        self.estimator_ = self._get_internal_estimator()
        self.estimator_.fit(X)

        mu = self.estimator_.mean_
        Sigma = self.estimator_.covariance_

        try:
            solution_dict = jl.solve_delage_dro(
                mu, Sigma, self.rho, self.k, self.p, self.solver_params
            )
            self.weights_ = np.asarray(solution_dict["weights"])
            self.objective_value_ = solution_dict["objective"]
            self.termination_status_ = solution_dict["status"]

        except Exception as e:
            print(f"An error occurred during the Julia optimization: {e}")
            self.weights_ = None
            self.objective_value_ = None
            self.termination_status_ = "JULIA_ERROR"
            raise e

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fit with {self.n_features_in_} features."
            )
        return X @ self.weights_

    def score(self, X, y=None):
        check_is_fitted(self)
        asset_returns = X[:, :-1]
        index_returns = X[:, -1]
        asset_weights = self.weights_[:-1]
        portfolio_returns = asset_returns @ asset_weights

        if self.p == 1:
            tracking_errors = np.abs(portfolio_returns - index_returns)
        elif self.p == 2:
            tracking_errors = (portfolio_returns - index_returns) ** 2
        else:
            raise ValueError("p must be 1 or 2.")
        mean_tracking_error = np.mean(tracking_errors)

        return -mean_tracking_error


class GelbrichOptimizer(BaseEstimator, RegressorMixin):
    """
    Solves a distributionally robust index tracking problem using JuMP and MOSEK.

    This optimizer finds a portfolio that performs well under the worst-case
    distribution within the Gelbrich ambiguity set defined by the L1 or L2 norm.

    Parameters
    ----------
    rho : float, default=0.1
        The radius of the Gelbrich ambiguity set.
        Determines the level of robustness.

    p : {1, 2}, default='2'
        The exponent used to define the loss function.

    solver_params : dict, optional
        Parameters to be passed to the JuMP/MOSEK solver. For example,
        to suppress solver output, you can pass `{'silent': True}`.
        JuMP's `set_silent(model)` is used by default.
    """

    def __init__(
        self,
        rho=0.1,
        p=2,
        estimator_type="empirical",
        estimator_params=None,
        solver_params={"silent": True},
    ):
        self.rho = rho
        self.p = p
        self.estimator_type = estimator_type
        self.estimator_params = estimator_params if estimator_params is not None else {}
        self.solver_params = solver_params if solver_params is not None else {}
        self.weights_ = None
        self.objective_value_ = None
        self.model_ = None
        self.termination_status_ = None

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
        Fit the index tracking model by solving the DRO problem.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Asset returns matrix, where n_samples is the number of time periods
            and n_features is the number of assets plus one.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)
        n_samples, n_features = X.shape
        n_assets = n_features - 1

        self.estimator_ = self._get_internal_estimator()
        self.estimator_.fit(X)

        mu = self.estimator_.mean_
        Sigma = self.estimator_.covariance_

        if self.rho <= 1e-6:
            weights, obj_val, status = jl.solve_chebyshev_dro(
                mu, Sigma, self.p, self.solver_params
            )
            self.weights_ = np.asarray(weights)
            self.objective_value_ = obj_val
            self.termination_status_ = status
            return self

        mode = self.solver_params.get("mode", "chol")
        if mode == "sqrt":
            weights, obj_val, status = jl.solve_gelbrich_dro_sqrt(
                mu, Sigma, self.rho, self.p, self.solver_params
            )
        elif mode == "chol":
            weights, obj_val, status = jl.solve_gelbrich_dro_chol(
                mu, Sigma, self.rho, self.p, self.solver_params
            )
        elif mode == "eig":
            weights, obj_val, status = jl.solve_gelbrich_dro_eig(
                mu, Sigma, self.rho, self.p, self.solver_params
            )
        elif mode == "eye":
            weights, obj_val, status = jl.solve_gelbrich_dro_eye(
                mu, Sigma, self.rho, self.p, self.solver_params
            )
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Valid modes are: 'sqrt', 'chol', 'eig'"
            )

        self.weights_ = np.asarray(weights)
        self.objective_value_ = obj_val
        self.termination_status_ = status

        return self

    def predict(self, X):
        """
        Calculates the portfolio returns for a given set of asset returns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Asset returns matrix.

        Returns
        -------
        portfolio_returns : array-like of shape (n_samples,)
            The calculated returns of the portfolio for each sample.
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fit with {self.n_features_in_} features."
            )

        return X @ self.weights_

    def score(self, X, y=None):
        check_is_fitted(self)
        asset_returns = X[:, :-1]
        index_returns = X[:, -1]
        asset_weights = self.weights_[:-1]
        portfolio_returns = asset_returns @ asset_weights

        if self.p == 1:
            tracking_errors = np.abs(portfolio_returns - index_returns)
        elif self.p == 2:
            tracking_errors = (portfolio_returns - index_returns) ** 2
        else:
            raise ValueError("p must be 1 or 2.")
        mean_tracking_error = np.mean(tracking_errors)

        return -mean_tracking_error