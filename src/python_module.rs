//! A wrapper to allow calling fbleau from Python.
//!
//! Wraps the function `fbleau_estimation::run_fbleau()`.
use numpy::*;
use pyo3::prelude::*;

use Label;
use estimates::*;
use fbleau_estimation::run_fbleau;

/// F-BLEAU is a tool for estimating the leakage of a system about its secrets
/// in a black-box manner (i.e., by only looking at examples of secret inputs
/// and respective outputs). It considers a generic system as a black-box,
/// taking secret inputs and returning outputs accordingly, and it measures
/// how much the outputs "leak" about the inputs.
#[pymodule(fbleau)]
fn pyfbleau(_py: Python, m: &PyModule) -> PyResult<()> {
    /// run_fbleau(train_x, train_y, test_x, test_y, estimate, knn_strategy,
    /// distance, logfile, delta, qstop, absolute, scale)
    /// --
    ///
    /// Run F-BLEAU for the chosen estimate.
    ///
    /// Keyword arguments:
    /// train_x : training observations
    /// train_y : training secrets
    /// test_x : test observations
    /// test_y : test secrets
    /// estimate : estimate, value in ("nn", "knn", "frequentist", "nn-bound")
    /// knn_strategy : if estimate is "knn", specify one in ("ln", "log10")
    /// distance : the distance used for NN or k-NN
    /// logfile : if specified, log the estimate values at every step in this file
    /// delta : use to stop fbleau when it reaches (delta, qstop)-convergence
    /// qstop : use to stop fbleau when it reaches (delta, qstop)-convergence
    /// absolute : measure absolute instead of relative convergence
    /// scale : scale observations' features in [0,1]
	#[pyfn(m, "run_fbleau")]
	fn run_fbleau_py(_py: Python,
                     train_x: &PyArray2<f64>, train_y: &PyArray1<Label>,
					 test_x: &PyArray2<f64>, test_y: &PyArray1<Label>,
					 estimate: &str, knn_strategy: Option<&str>,
					 distance: Option<String>, logfile: Option<String>,
					 delta: Option<f64>, qstop: Option<usize>, absolute: bool,
                     scale: bool)
			-> (f64, f64, f64) {

        // FIXME: Make run_fbleau() accept just a reference to them.
		let train_x = train_x.as_array().to_owned();
		let train_y = train_y.as_array().to_owned();
		let test_x = test_x.as_array().to_owned();
		let test_y = test_y.as_array().to_owned();

        // TODO: a more compact way to do the following is by
        // using serde's Deserialize (already implemented for Estimate
        // and KNNStrategy).
        let estimate = match estimate {
            "nn" => Estimate::NN,
            "knn" => Estimate::KNN,
            "frequentist" => Estimate::Frequentist,
            "nn-bound" => Estimate::NNBound,
            _ => { unimplemented!() },
        };
        let knn_strategy = if let Some(strategy) = knn_strategy {
            match strategy {
                "ln" => Some(KNNStrategy::Ln),
                "log10" => Some(KNNStrategy::Log10),
                _ => { unimplemented!() },
            }
        } else {
            None
        };

		run_fbleau(train_x, train_y, test_x, test_y, estimate, knn_strategy,
				   distance, logfile, delta, qstop, absolute, scale)
	}
	Ok(())
}
