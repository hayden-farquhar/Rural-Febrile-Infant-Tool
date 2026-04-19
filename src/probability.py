"""Bayesian IBI probability estimation from decision rule outputs.

Uses HSROC-pooled sensitivity/specificity and age-specific prevalence
to compute posterior IBI probability via Bayes' theorem.

P(IBI | rule-negative) = (1 - sens) × prev / [(1 - sens) × prev + spec × (1 - prev)]
P(IBI | rule-positive) = sens × prev / [sens × prev + (1 - spec) × (1 - prev)]

For multi-rule ensembles, we use an independence-weighted combination
of log-likelihood ratios (assuming conditional independence given IBI status).
"""

import numpy as np
from dataclasses import dataclass


# HSROC-pooled and single-study estimates from Phase 2
# (results/tables/hsroc_pooled_results.csv)
RULE_PERFORMANCE = {
    "Aronson": {"sens": 0.9148, "spec": 0.2798, "source": "pooled (2 cohorts)"},
    "PECARN": {"sens": 0.9552, "spec": 0.6101, "source": "pooled (2 cohorts)"},
    "AAP 2021": {"sens": 0.9851, "spec": 0.2269, "source": "FIDO 2024"},
    "BSAC": {"sens": 0.9851, "spec": 0.2018, "source": "FIDO 2024"},
    "Step-by-Step": {"sens": 0.9195, "spec": 0.4690, "source": "Gomez 2016"},
    "NICE_NG143": {"sens": 0.9254, "spec": 0.2737, "source": "FIDO 2024"},
    "Rochester": {"sens": 0.8161, "spec": 0.4447, "source": "Gomez 2016"},
}

# Age-specific IBI prevalence from published cohorts
AGE_PREVALENCE = {
    "0-28d": 0.035,   # Kuppermann 2019, Aronson 2019
    "29-60d": 0.015,  # Kuppermann 2019
    "61-89d": 0.020,  # Aronson 2025
}


def get_prevalence(age_days: int) -> float:
    if age_days <= 28:
        return AGE_PREVALENCE["0-28d"]
    elif age_days <= 60:
        return AGE_PREVALENCE["29-60d"]
    else:
        return AGE_PREVALENCE["61-89d"]


def posterior_ibi_single_rule(
    rule_positive: bool,
    sens: float,
    spec: float,
    prior: float,
) -> float:
    """Compute posterior P(IBI) given a single rule result via Bayes' theorem."""
    if rule_positive:
        # P(IBI | rule+) = sens × prior / [sens × prior + (1-spec) × (1-prior)]
        numerator = sens * prior
        denominator = sens * prior + (1 - spec) * (1 - prior)
    else:
        # P(IBI | rule-) = (1-sens) × prior / [(1-sens) × prior + spec × (1-prior)]
        numerator = (1 - sens) * prior
        denominator = (1 - sens) * prior + spec * (1 - prior)

    if denominator == 0:
        return prior
    return numerator / denominator


def posterior_ibi_multi_rule(
    rule_results: dict,
    age_days: int,
) -> tuple[float, dict]:
    """Compute posterior P(IBI) from multiple rule results.

    Uses log-likelihood ratio combination assuming conditional independence.
    This is a simplifying assumption — rules share inputs and are NOT truly
    independent. The manuscript should note this.

    Args:
        rule_results: dict of {rule_name: rule_result} from apply functions
        age_days: patient age in days

    Returns:
        (posterior_probability, per_rule_details)
    """
    prior = get_prevalence(age_days)
    log_prior_odds = np.log(prior / (1 - prior))

    details = {}
    total_log_lr = 0.0
    n_contributing = 0

    for name, result in rule_results.items():
        if not result.applicable:
            details[name] = {"status": "not_applicable", "lr": None, "posterior": None}
            continue

        perf = RULE_PERFORMANCE.get(name)
        if perf is None:
            details[name] = {"status": "no_performance_data", "lr": None, "posterior": None}
            continue

        sens = perf["sens"]
        spec = perf["spec"]

        if result.prediction == 1:  # rule-positive (not low-risk)
            lr = sens / (1 - spec)
        else:  # rule-negative (low-risk)
            lr = (1 - sens) / spec

        # Per-rule posterior (for display)
        per_rule_post = posterior_ibi_single_rule(
            result.prediction == 1, sens, spec, prior
        )

        details[name] = {
            "status": "positive" if result.prediction == 1 else "negative",
            "lr": lr,
            "posterior": per_rule_post,
            "sens": sens,
            "spec": spec,
        }

        # Weight the LR contribution — downweight correlated rules
        # Rules sharing inputs get reduced weight to partially account
        # for non-independence
        weight = _correlation_weight(name, rule_results)
        total_log_lr += weight * np.log(lr)
        n_contributing += 1

    if n_contributing == 0:
        return prior, details

    # Combine: posterior odds = prior odds × product of (weighted) LRs
    log_posterior_odds = log_prior_odds + total_log_lr
    posterior_odds = np.exp(np.clip(log_posterior_odds, -20, 20))
    posterior = posterior_odds / (1 + posterior_odds)

    # Also identify the best single-rule estimate (most informative)
    # The rule with the highest specificity gives the most informative
    # "low-risk" classification (smallest posterior when negative)
    best_rule = None
    best_posterior = prior
    for name, detail in details.items():
        if detail["status"] == "negative" and detail["posterior"] is not None:
            if detail["posterior"] < best_posterior:
                best_posterior = detail["posterior"]
                best_rule = name
        elif detail["status"] == "positive" and detail["posterior"] is not None:
            if detail["posterior"] > best_posterior:
                best_posterior = detail["posterior"]
                best_rule = name

    # Compute uncertainty interval from per-rule posterior range
    applicable_posteriors = [
        d["posterior"] for d in details.values()
        if d["posterior"] is not None and d["status"] != "not_applicable"
           and d["status"] != "no_performance_data"
    ]
    if applicable_posteriors:
        interval_lo = min(applicable_posteriors)
        interval_hi = max(applicable_posteriors)
    else:
        interval_lo = prior
        interval_hi = prior

    return float(posterior), details, best_rule, float(best_posterior), (interval_lo, interval_hi)


def _correlation_weight(rule_name: str, all_results: dict) -> float:
    """Downweight correlated rules to partially account for non-independence.

    Rules that share the same key inputs (e.g., ANC-based rules) get
    reduced weight when combined with other ANC-based rules.
    """
    # Input overlap groups
    anc_rules = {"Aronson", "PECARN", "AAP 2021", "Step-by-Step"}
    pct_crp_rules = {"PECARN", "AAP 2021", "Step-by-Step"}
    ua_rules = {"Aronson", "PECARN", "AAP 2021", "Step-by-Step", "Rochester"}

    applicable_names = {n for n, r in all_results.items() if r.applicable}
    n_applicable = len(applicable_names)

    if n_applicable <= 1:
        return 1.0

    # Count how many other applicable rules share input groups with this rule
    shared = 0
    if rule_name in anc_rules:
        shared += len(anc_rules & applicable_names) - 1
    if rule_name in pct_crp_rules:
        shared += len(pct_crp_rules & applicable_names) - 1
    if rule_name in ua_rules:
        shared += len(ua_rules & applicable_names) - 1

    # Weight: 1/sqrt(1 + shared_count) — diminishing returns for correlated rules
    return 1.0 / np.sqrt(1 + shared)
