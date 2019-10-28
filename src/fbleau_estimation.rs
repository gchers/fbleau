//! F-BLEAU estimation routines.

/// Main estimation routine.
///
/// Given training and evaluation (test) data, this function runs
/// the desired estimator and returns a tuple containing:
///     - smallest estimate
///     - final estimate (i.e., the estimate when all the training data was
///       available)
///     - random guessing error.
use ndarray::*;
use std::fs::File;
use std::io::Write;

use fbleau::Label;
use fbleau::estimates::*;
use utils::{prepare_data, estimate_random_guessing,has_integer_support};


pub fn run_fbleau(train_x: Array2<f64>, train_y: Array1<Label>,
                  test_x: Array2<f64>, test_y: Array1<Label>,
                  estimate: Estimate, knn_strategy: Option<KNNStrategy>,
                  distance: Option<String>, logfile: Option<String>,
                  delta: Option<f64>, qstop: Option<usize>, absolute: bool,
                  scale: bool)
        -> (f64, f64, f64) {

    // Check label's indexes, and scale data if required.
    let (train_x, train_y, test_x, test_y, nlabels) =
        prepare_data(train_x, train_y, test_x, test_y, scale);

    // Convergence with (delta, q)-convergence checker.
    let convergence_checker = if let Some(delta) = delta {
        let q = match qstop {
            Some(q) => q,
            None => (train_x.len() as f64 * 0.1) as usize,
        };
        println!("will stop when (delta={}, q={})-converged", delta, q);
        Some(ForwardChecker::new(&[delta], q, !absolute))
    } else if qstop.is_some() {
        panic!("--qstop should only be specified with --delta");
    } else {
        // No convergence checker (i.e., run for all training data).
        None
    };

    // Random guessing error.
    let random_guessing = estimate_random_guessing(&test_y.view());
    println!("Random guessing error: {}", random_guessing);
    println!("Estimating leakage measures...");

    // Distance for k-NN (defaults to Euclidean).
    let distance = match distance.as_ref().map(String::as_ref) {
        Some("euclidean") => euclidean_distance,
        Some("levenshtein") => levenshtein_distance,
        _ => euclidean_distance,
    };

    // Init estimator and run.
    let (min_error, last_error) = match estimate {
        Estimate::Frequentist => {
            if !has_integer_support(&train_x) || !has_integer_support(&test_x) {
                println!("Warning: frequentist discouraged for continuous observations!");
            }
            let estimator = FrequentistEstimator::new(nlabels,
                                                      &test_x.view(),
                                                      &test_y.view());
            run_forward_strategy(estimator, convergence_checker, logfile,
                                 train_x, train_y)
            },
        Estimate::NN => {
            if !has_integer_support(&train_x) || !has_integer_support(&test_x) {
                println!("Warning: NN discouraged for continuous observations!");
            }
            let estimator = KNNEstimator::new(&test_x.view(), &test_y.view(),
                                              train_x.nrows(), distance,
                                              KNNStrategy::NN);
            run_forward_strategy(estimator, convergence_checker, logfile,
                                 train_x, train_y)
            },
        Estimate::KNN => {
            let estimator = KNNEstimator::new(&test_x.view(), &test_y.view(),
                                              train_x.nrows(), distance,
                                              knn_strategy.expect(
                                                  "Specify a k-NN strategy."));
            run_forward_strategy(estimator, convergence_checker, logfile,
                                 train_x, train_y)
            },
        Estimate::NNBound => {
            let estimator = NNBoundEstimator::new(&test_x.view(), &test_y.view(),
                                                  distance, nlabels);
            run_forward_strategy(estimator, convergence_checker, logfile,
                                 train_x, train_y)
            },
    };
    (min_error, last_error, random_guessing)
}

/// Forward strategy for estimation.
///
/// Estimates security measures with a forward strategy: the estimator
/// is trained with an increasing number of examples, and its estimate
/// is progressively logged.
/// This function returns:
///     - smallest estimate
///     - final estimate (i.e., the estimate when all the training data was
///       available).
fn run_forward_strategy<E>(mut estimator: E,
                           mut convergence_checker: Option<ForwardChecker>,
                           logfile: Option<String>, train_x: Array2<f64>,
                           train_y: Array1<Label>) -> (f64, f64)
where E: BayesEstimator {
    // Init log file.
    let mut logfile = if let Some(fname) = logfile {
        let mut logfile = File::create(&fname)
                               .expect("couldn't open file for verbose logging");
        writeln!(logfile, "n, error-count, estimate")
            .expect("failed to write to verbose file");
        Some(logfile)
    } else {
        None
    };

    // We keep track both of the minimum and of the last estimate.
    let mut min_error = 1.0;
    let mut last_error = 1.0;

    for (n, (x, y)) in train_x.outer_iter().zip(train_y.iter()).enumerate() {
        // Compute error.
        estimator.add_example(&x, *y)
                 .expect("Could not add more examples.");
        last_error = estimator.get_error();

        if min_error > last_error {
            min_error = last_error;
        }

        if let Some(ref mut logfile) = logfile {
            writeln!(logfile, "{}, {}, {}", n, estimator.get_error_count(),
                     last_error).expect("failed to write to verbose file");
        }

        // Should we stop because of (delta, q)-convergence?
        if let Some(ref mut checker) = convergence_checker {
            checker.add_estimate(last_error);
            if checker.all_converged() {
                break;
            }
        }
    }

    (min_error, last_error)
}
