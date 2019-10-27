//! An estimator returning the bound based on the NN classifier.
use ndarray::*;

use Label;
use estimates::{BayesEstimator,KNNEstimator,KNNStrategy,nn_bound};

/// Defines an estimator that returns the NN bound by Cover&Hart.
///
/// This estimate is asymptotically guaranteed to lower bound the
/// true Bayes risk.
pub struct NNBoundEstimator<D>
where D: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64 + Send + Sync + Copy {
    knn: KNNEstimator<D>,
    nlabels: usize,
}

impl<D> NNBoundEstimator<D>
where D: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64 + Send + Sync + Copy {
    /// Create a new NN bound estimator.
    pub fn new(test_x: &ArrayView2<f64>, test_y: &ArrayView1<Label>,
               distance: D, nlabels: usize) -> NNBoundEstimator<D> {
        
        // NOTE: the value of max_n here does not matter, as it is
        // only used for computing max_k, which is fixed to 1
        // for the KNNStrategy:NN.
        let max_n = 1;

        NNBoundEstimator {
            knn: KNNEstimator::new(test_x, test_y, max_n, distance,
                                   KNNStrategy::NN),
            nlabels: nlabels,
        }
    }
}

/// This implementation maps exactly that of KNNEstimator,
/// except for get_error(), which returns the bound.
impl<D> BayesEstimator for NNBoundEstimator<D>
where D: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64 + Send + Sync + Copy {
    /// Adds a new example.
    fn add_example(&mut self, x: &ArrayView1<f64>, y: Label) -> Result<(), ()> {
        self.knn.add_example(x, y)
    }
    /// Returns the error count.
    fn get_error_count(&self) -> usize {
        self.knn.get_error_count()
    }

    /// Returns the error for the current k.
    fn get_error(&self) -> f64 {
        let error = self.knn.get_error();
        nn_bound(error, self.nlabels)
    }
}

