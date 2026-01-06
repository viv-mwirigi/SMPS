"""
Sensitivity Analysis and Parameter Identifiability (Gap 11).

This module provides tools for understanding parameter importance
and uncertainty in the water balance model:

1. MORRIS SCREENING: Efficient identification of important parameters
2. SOBOL INDICES: Global sensitivity analysis with interaction effects
3. BAYESIAN CALIBRATION: MCMC-based posterior estimation
4. CROSS-VALIDATION: Leave-one-site-out transferability assessment

References:
- Morris (1991) Factorial Sampling Plans for Preliminary Computational Experiments
- Sobol (2001) Global sensitivity indices for nonlinear mathematical models
- Vrugt et al. (2003) DREAM algorithm for Bayesian calibration
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETER SPACE DEFINITION
# =============================================================================

@dataclass
class Parameter:
    """
    Definition of a model parameter for sensitivity analysis.
    """
    name: str
    min_value: float
    max_value: float
    default_value: float = None
    distribution: str = "uniform"  # uniform, normal, lognormal

    # For normal/lognormal
    mean: float = None
    std: float = None

    # Metadata
    units: str = ""
    description: str = ""

    def __post_init__(self):
        if self.default_value is None:
            self.default_value = (self.min_value + self.max_value) / 2

    def sample(self, n: int = 1, rng: np.random.Generator = None) -> np.ndarray:
        """Sample n values from this parameter's distribution"""
        if rng is None:
            rng = np.random.default_rng()

        if self.distribution == "uniform":
            return rng.uniform(self.min_value, self.max_value, n)
        elif self.distribution == "normal":
            mean = self.mean or self.default_value
            std = self.std or (self.max_value - self.min_value) / 6
            samples = rng.normal(mean, std, n)
            return np.clip(samples, self.min_value, self.max_value)
        elif self.distribution == "lognormal":
            mean = self.mean or np.log(self.default_value)
            std = self.std or 0.5
            samples = rng.lognormal(mean, std, n)
            return np.clip(samples, self.min_value, self.max_value)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class ParameterSpace:
    """Collection of parameters defining the model's parameter space"""
    parameters: List[Parameter] = field(default_factory=list)

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, key: str) -> Parameter:
        for p in self.parameters:
            if p.name == key:
                return p
        raise KeyError(f"Parameter '{key}' not found")

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.parameters]

    @property
    def bounds(self) -> np.ndarray:
        """Return bounds as (n_params, 2) array"""
        return np.array([[p.min_value, p.max_value] for p in self.parameters])

    @property
    def defaults(self) -> np.ndarray:
        """Return default values"""
        return np.array([p.default_value for p in self.parameters])

    def sample(self, n: int, rng: np.random.Generator = None) -> np.ndarray:
        """Sample n parameter sets from the space"""
        if rng is None:
            rng = np.random.default_rng()

        samples = np.column_stack([
            p.sample(n, rng) for p in self.parameters
        ])
        return samples

    def to_dict(self, values: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary"""
        return {p.name: values[i] for i, p in enumerate(self.parameters)}


# Default parameter space for water balance model
def create_default_parameter_space() -> ParameterSpace:
    """Create default parameter space for water balance model"""
    return ParameterSpace([
        Parameter("theta_r", 0.02, 0.15, 0.05, units="m³/m³",
                  description="Residual water content"),
        Parameter("theta_s", 0.35, 0.55, 0.45, units="m³/m³",
                  description="Saturated water content"),
        Parameter("alpha", 0.001, 0.1, 0.02, units="1/cm",
                  description="Van Genuchten alpha"),
        Parameter("n", 1.1, 3.0, 1.5, units="-",
                  description="Van Genuchten n"),
        Parameter("Ksat", 0.1, 100, 10, units="cm/day",
                  description="Saturated hydraulic conductivity"),
        Parameter("Kcb", 0.3, 1.5, 0.9, units="-",
                  description="Basal crop coefficient"),
        Parameter("root_depth", 0.2, 2.0, 0.6, units="m",
                  description="Root zone depth"),
        Parameter("wilting_point_psi", -15000, -5000, -15000, units="cm",
                  description="Wilting point pressure head"),
        Parameter("field_capacity_psi", -500, -100, -340, units="cm",
                  description="Field capacity pressure head"),
    ])


# =============================================================================
# MORRIS SCREENING
# =============================================================================

@dataclass
class MorrisResults:
    """
    Results from Morris screening analysis.

    Interpretation:
    - mu_star: Overall importance of parameter (mean of absolute elementary effects)
    - sigma: Interactions/non-linearity (std of elementary effects)
    - mu_star/sigma > 1: Mostly additive effect
    - mu_star/sigma < 1: Strong interactions
    """
    parameter_names: List[str]
    mu: np.ndarray       # Mean of elementary effects
    mu_star: np.ndarray  # Mean of absolute elementary effects
    sigma: np.ndarray    # Std of elementary effects

    # Raw data
    elementary_effects: np.ndarray  # (n_trajectories, n_params)

    def ranking(self) -> List[str]:
        """Return parameters ranked by importance (mu_star)"""
        order = np.argsort(self.mu_star)[::-1]
        return [self.parameter_names[i] for i in order]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        return pd.DataFrame({
            'Parameter': self.parameter_names,
            'mu': self.mu,
            'mu_star': self.mu_star,
            'sigma': self.sigma,
            'mu_star/sigma': self.mu_star / (self.sigma + 1e-10)
        }).sort_values('mu_star', ascending=False)


def morris_screening(
    model_func: Callable[[np.ndarray], float],
    param_space: ParameterSpace,
    n_trajectories: int = 10,
    n_levels: int = 4,
    seed: int = 42
) -> MorrisResults:
    """
    Morris screening for parameter importance.

    This is an efficient one-at-a-time (OAT) method that requires only
    r*(k+1) model evaluations for r trajectories and k parameters.

    Args:
        model_func: Function that takes parameter array and returns scalar output
        param_space: ParameterSpace defining parameter bounds
        n_trajectories: Number of random trajectories (typically 10-20)
        n_levels: Number of grid levels (typically 4-6)
        seed: Random seed for reproducibility

    Returns:
        MorrisResults with sensitivity indices
    """
    rng = np.random.default_rng(seed)
    k = len(param_space)

    # Grid step
    delta = n_levels / (2 * (n_levels - 1))

    elementary_effects = np.zeros((n_trajectories, k))

    for r in range(n_trajectories):
        # Generate trajectory
        trajectory = _generate_morris_trajectory(k, n_levels, delta, rng)

        # Scale to actual parameter bounds
        bounds = param_space.bounds
        trajectory_scaled = bounds[:, 0] + \
            trajectory * (bounds[:, 1] - bounds[:, 0])

        # Evaluate model at each point
        outputs = np.array([model_func(x) for x in trajectory_scaled])

        # Compute elementary effects
        for i in range(k):
            # Find where parameter i changed
            for j in range(k):
                if not np.allclose(trajectory[j+1], trajectory[j]):
                    param_changed = np.argmax(
                        np.abs(trajectory[j+1] - trajectory[j])
                    )
                    if param_changed == i:
                        ef = (outputs[j+1] - outputs[j]) / delta
                        elementary_effects[r, i] = ef
                        break

    # Compute statistics
    mu = np.mean(elementary_effects, axis=0)
    mu_star = np.mean(np.abs(elementary_effects), axis=0)
    sigma = np.std(elementary_effects, axis=0)

    return MorrisResults(
        parameter_names=param_space.names,
        mu=mu,
        mu_star=mu_star,
        sigma=sigma,
        elementary_effects=elementary_effects
    )


def _generate_morris_trajectory(
    k: int,
    n_levels: int,
    delta: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Generate one Morris trajectory through parameter space"""
    # Start point (random grid point)
    x = rng.choice(n_levels, k) / (n_levels - 1)

    # Ensure we can add delta in random direction
    x = np.where(x > 0.5, x - delta, x)

    trajectory = [x.copy()]

    # Random permutation of parameter indices
    order = rng.permutation(k)

    # Step through each parameter
    for i in order:
        x_new = x.copy()
        # Random direction
        direction = rng.choice([-1, 1])
        x_new[i] = x[i] + direction * delta

        # Ensure bounds
        if x_new[i] < 0:
            x_new[i] = x[i] + delta
        elif x_new[i] > 1:
            x_new[i] = x[i] - delta

        trajectory.append(x_new.copy())
        x = x_new

    return np.array(trajectory)


# =============================================================================
# SOBOL SENSITIVITY ANALYSIS
# =============================================================================

@dataclass
class SobolResults:
    """
    Results from Sobol sensitivity analysis.

    S1: First-order index (main effect)
    ST: Total-order index (includes all interactions)
    S2: Second-order indices (pairwise interactions)

    Interpretation:
    - Sum(S1) ≈ 1: Model is additive
    - ST - S1: Contribution from interactions
    - S1 close to ST: No significant interactions for this parameter
    """
    parameter_names: List[str]
    S1: np.ndarray           # First-order indices
    S1_conf: np.ndarray      # Confidence intervals
    ST: np.ndarray           # Total-order indices
    ST_conf: np.ndarray      # Confidence intervals
    S2: Optional[np.ndarray] = None  # Second-order indices (k x k)

    # Diagnostic
    n_samples: int = 0
    converged: bool = False

    def ranking(self, use_total: bool = True) -> List[str]:
        """Return parameters ranked by importance"""
        indices = self.ST if use_total else self.S1
        order = np.argsort(indices)[::-1]
        return [self.parameter_names[i] for i in order]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        return pd.DataFrame({
            'Parameter': self.parameter_names,
            'S1': self.S1,
            'S1_conf': self.S1_conf,
            'ST': self.ST,
            'ST_conf': self.ST_conf,
            'Interactions': self.ST - self.S1
        }).sort_values('ST', ascending=False)


def sobol_analysis(
    model_func: Callable[[np.ndarray], float],
    param_space: ParameterSpace,
    n_samples: int = 1024,
    calc_second_order: bool = False,
    n_bootstrap: int = 100,
    seed: int = 42
) -> SobolResults:
    """
    Sobol global sensitivity analysis.

    Uses Saltelli's extension of Sobol's method to compute
    first-order and total-order sensitivity indices.

    Args:
        model_func: Function that takes parameter array and returns scalar
        param_space: ParameterSpace defining bounds
        n_samples: Base sample size (actual = n_samples * (2k + 2))
        calc_second_order: Whether to compute S2 (increases cost)
        n_bootstrap: Bootstrap samples for confidence intervals
        seed: Random seed

    Returns:
        SobolResults with sensitivity indices
    """
    rng = np.random.default_rng(seed)
    k = len(param_space)
    bounds = param_space.bounds

    # Generate Sobol sequence matrices A, B
    # Using Saltelli's scheme
    n_total = n_samples * \
        (2 * k + 2) if calc_second_order else n_samples * (k + 2)

    # Sample matrices
    A = _quasi_random_samples(n_samples, k, rng)
    B = _quasi_random_samples(n_samples, k, rng)

    # Scale to bounds
    A_scaled = bounds[:, 0] + A * (bounds[:, 1] - bounds[:, 0])
    B_scaled = bounds[:, 0] + B * (bounds[:, 1] - bounds[:, 0])

    # Evaluate base matrices
    logger.info("Evaluating %d base samples...", 2 * n_samples)
    y_A = np.array([model_func(x) for x in A_scaled])
    y_B = np.array([model_func(x) for x in B_scaled])

    # Evaluate AB matrices (A with columns from B)
    y_AB = np.zeros((k, n_samples))

    for i in range(k):
        AB = A_scaled.copy()
        AB[:, i] = B_scaled[:, i]
        logger.info("Evaluating %d samples for parameter %d/%d...",
                    n_samples, i+1, k)
        y_AB[i] = np.array([model_func(x) for x in AB])

    # Compute indices
    S1 = np.zeros(k)
    ST = np.zeros(k)

    f0_sq = np.mean(y_A) * np.mean(y_B)
    var_total = np.var(np.concatenate([y_A, y_B]))

    for i in range(k):
        # First-order
        S1[i] = np.mean(y_B * (y_AB[i] - y_A)) / var_total

        # Total-order
        ST[i] = np.mean((y_A - y_AB[i])**2) / (2 * var_total)

    # Bootstrap confidence intervals
    S1_conf = np.zeros(k)
    ST_conf = np.zeros(k)

    for i in range(k):
        s1_boot = []
        st_boot = []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_samples, n_samples)

            y_A_b = y_A[idx]
            y_B_b = y_B[idx]
            y_AB_b = y_AB[i, idx]

            var_b = np.var(np.concatenate([y_A_b, y_B_b]))
            if var_b > 1e-10:
                s1_boot.append(np.mean(y_B_b * (y_AB_b - y_A_b)) / var_b)
                st_boot.append(np.mean((y_A_b - y_AB_b)**2) / (2 * var_b))

        S1_conf[i] = 1.96 * np.std(s1_boot)
        ST_conf[i] = 1.96 * np.std(st_boot)

    # Second-order indices (optional)
    S2 = None
    if calc_second_order:
        S2 = _compute_second_order(y_A, y_B, y_AB, var_total)

    # Check convergence
    converged = np.all(S1_conf < 0.1 * np.abs(S1 + 1e-6))

    return SobolResults(
        parameter_names=param_space.names,
        S1=S1,
        S1_conf=S1_conf,
        ST=ST,
        ST_conf=ST_conf,
        S2=S2,
        n_samples=n_samples,
        converged=converged
    )


def _quasi_random_samples(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Generate quasi-random samples using Sobol sequence if available, else uniform"""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=k, scramble=True, seed=int(rng.integers(1e9)))
        return sampler.random(n)
    except ImportError:
        # Fallback to uniform random
        return rng.uniform(0, 1, (n, k))


def _compute_second_order(
    y_A: np.ndarray,
    y_B: np.ndarray,
    y_AB: np.ndarray,
    var_total: float
) -> np.ndarray:
    """Compute second-order Sobol indices"""
    k = len(y_AB)
    S2 = np.zeros((k, k))

    for i in range(k):
        for j in range(i + 1, k):
            # Second-order index S_ij
            # This requires additional AB_ij matrices which we don't have
            # in the basic implementation - set to NaN
            S2[i, j] = np.nan
            S2[j, i] = np.nan

    return S2


# =============================================================================
# BAYESIAN CALIBRATION WITH MCMC
# =============================================================================

@dataclass
class MCMCResults:
    """
    Results from MCMC Bayesian calibration.
    """
    parameter_names: List[str]

    # Posterior samples
    samples: np.ndarray  # (n_samples, n_params)
    log_likelihoods: np.ndarray

    # Diagnostics
    acceptance_rate: float
    effective_sample_size: np.ndarray
    gelman_rubin: Optional[np.ndarray] = None

    # Summary statistics
    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.samples, axis=0)

    @property
    def std(self) -> np.ndarray:
        return np.std(self.samples, axis=0)

    @property
    def median(self) -> np.ndarray:
        return np.median(self.samples, axis=0)

    def quantile(self, q: float) -> np.ndarray:
        return np.quantile(self.samples, q, axis=0)

    @property
    def ci_95(self) -> Tuple[np.ndarray, np.ndarray]:
        """95% credible intervals"""
        return self.quantile(0.025), self.quantile(0.975)

    def to_dataframe(self) -> pd.DataFrame:
        """Summary statistics as DataFrame"""
        ci_low, ci_high = self.ci_95
        return pd.DataFrame({
            'Parameter': self.parameter_names,
            'Mean': self.mean,
            'Std': self.std,
            'Median': self.median,
            'CI_2.5%': ci_low,
            'CI_97.5%': ci_high
        })

    @property
    def converged(self) -> bool:
        """Check if MCMC has converged (basic heuristics)"""
        if self.gelman_rubin is not None:
            return np.all(self.gelman_rubin < 1.1)
        return self.acceptance_rate > 0.1 and np.all(self.effective_sample_size > 100)


class MCMC:
    """
    Metropolis-Hastings MCMC for Bayesian calibration.

    Uses adaptive proposal distribution (Roberts & Rosenthal, 2009).
    """

    def __init__(
        self,
        log_likelihood: Callable[[np.ndarray], float],
        param_space: ParameterSpace,
        log_prior: Optional[Callable[[np.ndarray], float]] = None
    ):
        """
        Args:
            log_likelihood: Function returning log-likelihood for parameters
            param_space: ParameterSpace with bounds
            log_prior: Optional log-prior function (uniform if not provided)
        """
        self.log_likelihood = log_likelihood
        self.param_space = param_space
        self.log_prior = log_prior or self._uniform_prior

        self.k = len(param_space)
        self.bounds = param_space.bounds

    def _uniform_prior(self, theta: np.ndarray) -> float:
        """Uniform prior within bounds"""
        if np.all(theta >= self.bounds[:, 0]) and np.all(theta <= self.bounds[:, 1]):
            return 0.0  # log(1) = 0
        return -np.inf

    def sample(
        self,
        n_samples: int = 5000,
        n_burn: int = 1000,
        n_chains: int = 3,
        initial_cov_scale: float = 0.01,
        adapt_interval: int = 100,
        seed: int = 42
    ) -> MCMCResults:
        """
        Run MCMC sampling.

        Args:
            n_samples: Number of samples to keep per chain
            n_burn: Burn-in samples to discard
            n_chains: Number of parallel chains
            initial_cov_scale: Initial proposal covariance scale
            adapt_interval: Steps between covariance adaptations
            seed: Random seed

        Returns:
            MCMCResults with posterior samples
        """
        rng = np.random.default_rng(seed)

        # Storage for all chains
        all_samples = []
        all_logliks = []
        accept_counts = []

        for chain in range(n_chains):
            logger.info("Running chain %d/%d...", chain + 1, n_chains)

            samples, logliks, n_accept = self._run_chain(
                n_samples + n_burn,
                initial_cov_scale,
                adapt_interval,
                rng
            )

            # Discard burn-in
            all_samples.append(samples[n_burn:])
            all_logliks.append(logliks[n_burn:])
            accept_counts.append(n_accept)

        # Combine chains
        samples = np.vstack(all_samples)
        logliks = np.concatenate(all_logliks)

        # Compute diagnostics
        acceptance_rate = np.mean(accept_counts) / (n_samples + n_burn)
        ess = self._effective_sample_size(samples)

        # Gelman-Rubin if multiple chains
        gr = None
        if n_chains > 1:
            gr = self._gelman_rubin(all_samples)

        return MCMCResults(
            parameter_names=self.param_space.names,
            samples=samples,
            log_likelihoods=logliks,
            acceptance_rate=acceptance_rate,
            effective_sample_size=ess,
            gelman_rubin=gr
        )

    def _run_chain(
        self,
        n_iter: int,
        cov_scale: float,
        adapt_interval: int,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run single MCMC chain"""
        # Initialize
        theta = self.param_space.defaults.copy()

        # Add small perturbation
        theta += rng.normal(0, 0.01, self.k) * \
            (self.bounds[:, 1] - self.bounds[:, 0])
        theta = np.clip(theta, self.bounds[:, 0], self.bounds[:, 1])

        log_post = self.log_likelihood(theta) + self.log_prior(theta)

        # Proposal covariance
        cov = cov_scale * np.diag((self.bounds[:, 1] - self.bounds[:, 0])**2)

        samples = np.zeros((n_iter, self.k))
        logliks = np.zeros(n_iter)
        n_accept = 0

        # History for adaptation
        history = []

        for i in range(n_iter):
            # Propose
            theta_prop = rng.multivariate_normal(theta, cov)

            # Evaluate
            log_prior_prop = self.log_prior(theta_prop)

            if np.isfinite(log_prior_prop):
                log_lik_prop = self.log_likelihood(theta_prop)
                log_post_prop = log_lik_prop + log_prior_prop
            else:
                log_post_prop = -np.inf

            # Accept/reject
            log_alpha = log_post_prop - log_post

            if np.log(rng.random()) < log_alpha:
                theta = theta_prop
                log_post = log_post_prop
                n_accept += 1

            samples[i] = theta
            logliks[i] = log_post
            history.append(theta.copy())

            # Adapt covariance
            if (i + 1) % adapt_interval == 0 and i > adapt_interval:
                hist_arr = np.array(history[-adapt_interval:])
                cov = 2.38**2 / self.k * \
                    np.cov(hist_arr.T) + 1e-6 * np.eye(self.k)

        return samples, logliks, n_accept

    def _effective_sample_size(self, samples: np.ndarray) -> np.ndarray:
        """Estimate effective sample size using autocorrelation"""
        n = len(samples)
        ess = np.zeros(self.k)

        for i in range(self.k):
            x = samples[:, i]
            x = x - np.mean(x)

            # Autocorrelation
            acf = np.correlate(x, x, mode='full')
            acf = acf[n-1:] / acf[n-1]

            # Find cutoff
            cutoff = np.argmax(acf < 0.05)
            if cutoff == 0:
                cutoff = len(acf)

            tau = 1 + 2 * np.sum(acf[1:min(cutoff, 100)])
            ess[i] = n / max(tau, 1)

        return ess

    def _gelman_rubin(self, chains: List[np.ndarray]) -> np.ndarray:
        """Compute Gelman-Rubin convergence diagnostic"""
        n_chains = len(chains)
        n = len(chains[0])
        k = chains[0].shape[1]

        gr = np.zeros(k)

        for i in range(k):
            chain_means = np.array([np.mean(c[:, i]) for c in chains])
            chain_vars = np.array([np.var(c[:, i]) for c in chains])

            overall_mean = np.mean(chain_means)

            B = n * np.var(chain_means)  # Between-chain variance
            W = np.mean(chain_vars)      # Within-chain variance

            var_est = (n - 1) / n * W + B / n
            gr[i] = np.sqrt(var_est / W) if W > 0 else 1.0

        return gr


def create_gaussian_likelihood(
    model_func: Callable[[np.ndarray], np.ndarray],
    obs: np.ndarray,
    obs_error: float = 0.02
) -> Callable[[np.ndarray], float]:
    """
    Create Gaussian log-likelihood function.

    Args:
        model_func: Function that takes parameters, returns predictions
        obs: Observed values
        obs_error: Standard deviation of observation error

    Returns:
        Log-likelihood function
    """
    def log_likelihood(theta: np.ndarray) -> float:
        try:
            pred = model_func(theta)

            if len(pred) != len(obs):
                return -np.inf

            # Handle NaN
            valid = ~(np.isnan(obs) | np.isnan(pred))
            if np.sum(valid) < 10:
                return -np.inf

            residuals = pred[valid] - obs[valid]
            n = np.sum(valid)

            log_lik = -0.5 * n * np.log(2 * np.pi * obs_error**2) - \
                0.5 * np.sum(residuals**2) / obs_error**2

            return log_lik

        except (ValueError, RuntimeError, TypeError) as e:
            logger.warning("Model evaluation failed: %s", e)
            return -np.inf

    return log_likelihood


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

@dataclass
class CrossValidationResults:
    """Results from leave-one-site-out cross-validation"""
    site_names: List[str]

    # Per-site metrics when left out
    rmse_when_left_out: List[float]
    bias_when_left_out: List[float]
    kge_when_left_out: List[float]

    # Per-site metrics when included
    rmse_when_included: List[float]

    # Summary
    @property
    def mean_rmse_left_out(self) -> float:
        return np.nanmean(self.rmse_when_left_out)

    @property
    def transferability_score(self) -> float:
        """
        Score indicating how well model transfers to unseen sites.
        Higher is better. Ratio of performance when left out vs included.
        """
        left = np.nanmean(self.rmse_when_left_out)
        incl = np.nanmean(self.rmse_when_included)

        if incl > 0:
            return incl / left  # Should be < 1
        return np.nan

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'Site': self.site_names,
            'RMSE_left_out': self.rmse_when_left_out,
            'Bias_left_out': self.bias_when_left_out,
            'KGE_left_out': self.kge_when_left_out,
            'RMSE_included': self.rmse_when_included
        })


def leave_one_site_out_cv(
    sites_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    calibrate_func: Callable[[List[Tuple[np.ndarray, np.ndarray]]], np.ndarray],
    predict_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> CrossValidationResults:
    """
    Leave-one-site-out cross-validation.

    For each site:
    1. Calibrate model on all other sites
    2. Evaluate on left-out site

    Args:
        sites_data: Dict mapping site name to (forcing, observations)
        calibrate_func: Function that takes list of (forcing, obs) and returns parameters
        predict_func: Function that takes (parameters, forcing) and returns predictions
        metric_funcs: Dict of metric functions (default: RMSE, Bias, KGE)

    Returns:
        CrossValidationResults
    """
    site_names = list(sites_data.keys())
    n_sites = len(site_names)

    rmse_left = []
    bias_left = []
    kge_left = []
    rmse_incl = []

    for i, left_out in enumerate(site_names):
        logger.info("Cross-validation fold %d/%d: leaving out %s",
                    i+1, n_sites, left_out)

        # Training data (all except left-out)
        train_data = [
            sites_data[s] for s in site_names if s != left_out
        ]

        # Calibrate on training sites
        params = calibrate_func(train_data)

        # Predict on left-out site
        forcing, obs = sites_data[left_out]
        pred = predict_func(params, forcing)

        # Compute metrics
        valid = ~(np.isnan(obs) | np.isnan(pred))
        obs_v = obs[valid]
        pred_v = pred[valid]

        if len(obs_v) > 10:
            rmse = np.sqrt(np.mean((pred_v - obs_v)**2))
            bias = np.mean(pred_v - obs_v)

            # KGE
            r = np.corrcoef(obs_v, pred_v)[0, 1]
            alpha = np.std(pred_v) / \
                np.std(obs_v) if np.std(obs_v) > 0 else np.nan
            beta = np.mean(pred_v) / \
                np.mean(obs_v) if np.mean(obs_v) != 0 else np.nan
            kge = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

            rmse_left.append(rmse)
            bias_left.append(bias)
            kge_left.append(kge)
        else:
            rmse_left.append(np.nan)
            bias_left.append(np.nan)
            kge_left.append(np.nan)

        # Also calibrate WITH this site and predict on it
        all_data = list(sites_data.values())
        params_all = calibrate_func(all_data)
        pred_all = predict_func(params_all, forcing)

        valid = ~(np.isnan(obs) | np.isnan(pred_all))
        if np.sum(valid) > 10:
            rmse_incl.append(
                np.sqrt(np.mean((pred_all[valid] - obs[valid])**2)))
        else:
            rmse_incl.append(np.nan)

    return CrossValidationResults(
        site_names=site_names,
        rmse_when_left_out=rmse_left,
        bias_when_left_out=bias_left,
        kge_when_left_out=kge_left,
        rmse_when_included=rmse_incl
    )


# =============================================================================
# PARAMETER IDENTIFIABILITY
# =============================================================================

@dataclass
class IdentifiabilityResults:
    """Results from parameter identifiability analysis"""
    parameter_names: List[str]

    # Collinearity
    condition_number: float
    collinearity_indices: np.ndarray  # Per-parameter

    # Correlation matrix
    correlation_matrix: np.ndarray

    # Identifiability classification
    identifiable: List[bool]

    def get_non_identifiable(self) -> List[str]:
        """Return list of non-identifiable parameters"""
        return [
            self.parameter_names[i]
            for i, ident in enumerate(self.identifiable)
            if not ident
        ]


def analyze_identifiability(
    jacobian: np.ndarray,
    parameter_names: List[str],
    collinearity_threshold: float = 15.0
) -> IdentifiabilityResults:
    """
    Analyze parameter identifiability from Jacobian.

    Uses collinearity analysis to determine which parameters
    can be uniquely identified from the data.

    Args:
        jacobian: Jacobian matrix (n_obs × n_params)
        parameter_names: Parameter names
        collinearity_threshold: Threshold for collinearity index

    Returns:
        IdentifiabilityResults
    """
    # Normalize columns
    col_norms = np.linalg.norm(jacobian, axis=0)
    J_norm = jacobian / (col_norms + 1e-10)

    # SVD
    U, s, Vt = np.linalg.svd(J_norm, full_matrices=False)

    # Condition number
    condition_number = s[0] / s[-1] if s[-1] > 1e-10 else np.inf

    # Collinearity indices
    k = len(parameter_names)
    col_indices = np.zeros(k)

    for i in range(k):
        # Collinearity index for parameter i
        # Based on Brun et al. (2001)
        v = Vt[:, i]
        col_indices[i] = 1 / np.sqrt(np.sum((v * s)**2) + 1e-10)

    # Correlation matrix
    JtJ = J_norm.T @ J_norm
    diag = np.sqrt(np.diag(JtJ))
    corr = JtJ / np.outer(diag, diag + 1e-10)

    # Identifiability
    identifiable = [ci < collinearity_threshold for ci in col_indices]

    return IdentifiabilityResults(
        parameter_names=parameter_names,
        condition_number=condition_number,
        collinearity_indices=col_indices,
        correlation_matrix=corr,
        identifiable=identifiable
    )


def compute_jacobian(
    model_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    param_space: ParameterSpace,
    h: float = 1e-4
) -> np.ndarray:
    """
    Compute Jacobian of model outputs with respect to parameters.

    Uses finite differences.

    Args:
        model_func: Model function returning predictions
        params: Current parameter values
        param_space: Parameter space (for scaling)
        h: Step size for finite differences

    Returns:
        Jacobian matrix (n_outputs × n_params)
    """
    y0 = model_func(params)
    n_out = len(y0)
    n_params = len(params)

    jacobian = np.zeros((n_out, n_params))
    bounds = param_space.bounds

    for i in range(n_params):
        # Scale step by parameter range
        hi = h * (bounds[i, 1] - bounds[i, 0])

        params_p = params.copy()
        params_p[i] += hi

        y_p = model_func(params_p)

        jacobian[:, i] = (y_p - y0) / hi

    return jacobian


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_sensitivity_analysis(
    model_func: Callable[[np.ndarray], float],
    param_space: ParameterSpace,
    method: str = "morris",
    **kwargs
) -> Union[MorrisResults, SobolResults]:
    """
    Run sensitivity analysis with specified method.

    Args:
        model_func: Model function
        param_space: Parameter space
        method: "morris" or "sobol"
        **kwargs: Method-specific arguments

    Returns:
        MorrisResults or SobolResults
    """
    if method.lower() == "morris":
        return morris_screening(model_func, param_space, **kwargs)
    elif method.lower() == "sobol":
        return sobol_analysis(model_func, param_space, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_sensitivity_results(results: Union[MorrisResults, SobolResults]) -> None:
    """Plot sensitivity analysis results (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    if isinstance(results, MorrisResults):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(results.mu_star, results.sigma)
        for i, name in enumerate(results.parameter_names):
            ax.annotate(name, (results.mu_star[i], results.sigma[i]))
        ax.set_xlabel('μ* (mean absolute effect)')
        ax.set_ylabel('σ (standard deviation)')
        ax.set_title('Morris Screening Results')

    elif isinstance(results, SobolResults):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(results.parameter_names))
        width = 0.35

        ax.bar(x - width/2, results.S1, width, label='S1 (first-order)',
               yerr=results.S1_conf, capsize=3)
        ax.bar(x + width/2, results.ST, width, label='ST (total-order)',
               yerr=results.ST_conf, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(results.parameter_names, rotation=45, ha='right')
        ax.set_ylabel('Sensitivity Index')
        ax.set_title('Sobol Sensitivity Indices')
        ax.legend()

    plt.tight_layout()
    plt.show()
