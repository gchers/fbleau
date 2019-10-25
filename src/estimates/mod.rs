//! This module implements Bayes risk estimates, and heuristics for
//! evaluating convergence.
pub mod knn;
pub mod knn_utils;
pub mod frequentist;
pub mod convergence;

pub use self::knn::KNNEstimator;
pub use self::frequentist::FrequentistEstimator;
pub use self::convergence::ForwardChecker;
pub use self::knn_utils::{KNNStrategy,knn_strategy,nn_bound};

use Label;
use ndarray::prelude::*;
use strsim::generic_levenshtein;


pub trait BayesEstimator {
    /// Adds a new training example.
    fn add_example(&mut self, x: &ArrayView1<f64>, y: Label) -> Result<(), ()>;
    /// Returns the current number of errors.
    fn get_error_count(&self) -> usize;
    /// Returns the current error rate.
    fn get_error(&self) -> f64;
}

fn some_or_error<T>(opt: Option<T>) -> Result<T, ()> {
    match opt {
        Some(x) => Ok(x),
        None => Err(()),
    }
}

/// Returns the Euclidean distance between two vectors of f64 values.
pub fn euclidean_distance(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {
    v1.iter()
      .zip(v2.iter())
      .map(|(x,y)| (x - y).powi(2))
      .sum::<f64>()
      .sqrt()
}

/// Returns the Levenshtein distance between two vectors of f64 values.
pub fn levenshtein_distance(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {
    generic_levenshtein(v1, v2) as f64
}
