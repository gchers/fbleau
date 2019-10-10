//! Definitions of security and leakage measures.
//! 
//! In the documentation, we write R* to indicate the Bayes risk,
//! and G to indicate the error of random guessing (i.e., 1 - max priors).
//!
//! # References
//! [1] M. S. Alvim et al. "Additive and multiplicative notions of leakage,
//!     and their capacities." CSF, 2014.
//! [2] G. Cherubin "Bayes, not Naïve: Security Bounds on Website
//!     Fingerprinting Defenses." PoPETS, 2017
extern crate float_cmp;

use self::float_cmp::approx_eq;

/// Computes the Multiplicative Leakage, as defined in [1].
pub fn multiplicative_leakage(bayes_risk: f64, random_guessing: f64) -> f64 {
    assert!(!approx_eq!(f64, random_guessing, 0.),
            "Random guessing error cannot be 0");

    (1.-bayes_risk) / (1.-random_guessing)
}

/// Computes the Additive Leakage, as defined in [1].
pub fn additive_leakage(bayes_risk: f64, random_guessing: f64) -> f64 {
    assert!(!approx_eq!(f64, random_guessing, 0.),
            "Random guessing error cannot be 0");

    random_guessing - bayes_risk
}

/// Computes the Bayes security measure, as defined in [2].
pub fn bayes_security_measure(bayes_risk: f64, random_guessing: f64) -> f64 {
    assert!(!approx_eq!(f64, random_guessing, 0.),
            "Random guessing error cannot be 0");

    bayes_risk / random_guessing
}

/// Computes the Min-entropy leakage.
pub fn min_entropy_leakage(bayes_risk: f64, random_guessing: f64) -> f64 {
    assert!(!approx_eq!(f64, random_guessing, 0.),
            "Random guessing error cannot be 0");

    - (1.-random_guessing).log(2.) + (1.-bayes_risk).log(2.)
}
