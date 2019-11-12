//! This module implements Bayes risk estimates, and heuristics for
//! evaluating convergence.
pub mod knn;
pub mod knn_utils;
pub mod nn_bound;
pub mod frequentist;
pub mod convergence;

pub use self::knn::KNNEstimator;
pub use self::nn_bound::NNBoundEstimator;
pub use self::frequentist::FrequentistEstimator;
pub use self::convergence::ForwardChecker;
pub use self::knn_utils::*;

use Label;
use ndarray::prelude::*;

/// Estimators that F-BLEAU currently provides.
#[derive(Deserialize)]
#[serde(rename_all="lowercase")]
pub enum Estimate {
    NN,
    KNN,
    Frequentist,
    #[serde(rename="nn-bound")]
    NNBound,
}

/// Every estimator should implement this trait.
pub trait BayesEstimator {
    /// Adds a new training example.
    fn add_example(&mut self, x: &ArrayView1<f64>, y: Label) -> Result<(), ()>;
    /// Returns the current number of errors.
    fn get_error_count(&self) -> usize;
    /// Returns the current error rate.
    fn get_error(&self) -> f64;
    /// Returns the current errors for each test point.
    /// `true` means error, `false` no error.
    fn get_individual_errors(&self) -> Vec<bool>;
}

fn some_or_error<T>(opt: Option<T>) -> Result<T, ()> {
    match opt {
        Some(x) => Ok(x),
        None => Err(()),
    }
}
