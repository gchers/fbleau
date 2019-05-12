pub mod knn;
pub mod frequentist;
pub mod convergence;

pub use self::knn::KNNEstimator;
pub use self::frequentist::FrequentistEstimator;
pub use self::convergence::ForwardChecker;

use Label;
use ndarray::prelude::*;

use strsim::generic_levenshtein;

pub enum Estimator<F>
where F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64 {
    KNN(KNNEstimator<F>, Box<Fn(usize) -> usize>),
    Frequentist(FrequentistEstimator),
}

impl<F> Estimator<F>
where F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64 + Send + Sync + Copy {
    /// Inserts a new example in the training data, and returns
    /// the current error estimate.
    pub fn next(&mut self, n: usize, x: &ArrayView1<f64>, y: Label, distance: F) -> Result<f64, ()> {
        match self {
            &mut Estimator::KNN(ref mut estimator, ref kn) => {
                estimator.set_k(kn(n))?;
                estimator.add_example(x, y, distance)?;
                Ok(estimator.get_error())
            },
            &mut Estimator::Frequentist(ref mut estimator) => {
                estimator.add_example(*x, y);
                Ok(estimator.get_error())
            },
        }
    }

    /// Returns the current error count.
    pub fn error_count(&self) -> usize {
        // TODO: let get_error_count() be part of a trait.
        match self {
            &Estimator::KNN(ref estimator, _) => estimator.get_error_count(),
            &Estimator::Frequentist(ref estimator) => estimator.get_error_count(),
        }
    }
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
