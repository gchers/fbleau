//! F-BLEAU is a tool for estimating the leakage of a system about its secrets
//! in a black-box manner (i.e., by only looking at examples of secret inputs
//! and respective outputs). It considers a generic system as a black-box,
//! taking secret inputs and returning outputs accordingly, and it measures
//! how much the outputs "leak" about the inputs.
//!
//! F-BLEAU is based on the equivalence between estimating the error of a
//! Machine Learning model of a specific class and the estimation of
//! information leakage [1,2,3].
//!
//! This code was also used for the experiments of [2] on the following
//! evaluations: Gowalla, e-passport, and side channel attack to finite field
//! exponentiation.
//!
//! # Getting started
//!
//! F-BLEAU takes as input CSV data containing examples of system's inputs
//! and outputs.
//! It currently requires two CSV files as input: a _training_ file and a
//! _validation_ (or _test_) file, such as:
//!
//!     0, 0.1, 2.43, 1.1
//!     1, 0.0, 1.22, 1.1
//!     1, 1.0, 1.02, 0.1
//!     ...
//!
//! where the first column specifies the secret, and the remaining ones
//! indicate the output vector.
//!
//! It runs a chosen method for estimating the Bayes risk (smallest probability
//! of error of an adversary at predicting a secret given the respective output),
//! and relative security measures.
//!
//! The general syntax is:
//!
//!     fbleau <estimate> [options] <train> <test>
//!
//! ## Estimates
//!
//! Currently available estimates:
//!
//! **log** k-NN estimate, with `k = ln(n)`, where `n` is the number of training
//! examples.
//!
//! **log 10** k-NN estimate, with `k = log10(n)`, where `n` is the number of
//! training examples.
//!
//! **frequentist** (or "lookup table") Standard estimate. Note that this
//! is only applicable when the outputs are finite; also, it does not scale
//! well to large systems (e.g., large input/output spaces).
//!
//! Bounds and other estimates:
//!
//! **nn-bound** Produces a lower bound of R* discovered by Cover and Hard ('67),
//! which is based on the error of the NN classifier (1-NN).
//!
//! **--knn** Runs the k-NN classifier for a fixed k to be specified.
//! Note that this _does not_ guarantee convergence to the Bayes risk.
//!
//! ## Further options
//!
//! By default, `fbleau` runs until a convergence criterion is met.
//! We usually declare convergence if an estimate did not vary more
//! than `--delta`, either in relative (default) or absolute
//! (`--absolute`) value, from its value in the last `q` examples
//! (where `q` is specified with `--qstop`).
//! One can specify more than one deltas as comma-separated values, e.g.:
//! `--delta=0.1,0.01,0.001`.
//!
//! Optionally, one may choose to let the estimator run for all the training
//! set (`--run-all`), in which case `fbleau` will still report how many
//! examples where required for convergence.
//!
//! When the system's outputs are vectors, `fbleau` by default does not
//! scale their values. The option `--scale` allows scaling in 0-1.
extern crate ndarray;
extern crate docopt;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate itertools;
extern crate ndarray_parallel;

extern crate fbleau;

mod utils;
mod security_measures;

use ndarray::*;
use std::fs::File;
use docopt::Docopt;
use std::io::Write;
use ndarray_parallel::rayon;

use fbleau::Label;
use fbleau::estimates::*;
use security_measures::*;
use utils::{load_data, vectors_to_ids, scale01, estimate_random_guessing};


const USAGE: &'static str = "
Estimate k-NN error and convergence.

Usage: fbleau log [options] <train> <test>
       fbleau log10 [options] <train> <test>
       fbleau nn-bound [options] <train> <test>
       fbleau --knn=<k> [options] <train> <test>
       fbleau frequentist [options] <train> <test>
       fbleau (--help | --version)

Options:
    --verbose=<fname>           Logs estimates at each step.
    --delta=<d>                 Delta for delta covergence.
    --qstop=<q>                 Number of examples to declare
                                delta-convergence. Default is 10% of
                                training data.
    --absolute                  Use absolute convergence instead of relative
                                convergence.
    --max-k=<k>                 Number of neighbors to store, initially,
                                for each test point. May improve performances,
                                but only use if you know what you are doing.
    --scale                     Scale features before running k-NN
                                (only makes sense for objects of 2 or more
                                dimensions).
    --nprocs=<n>                Number of threads to spawn. By default it is
                                the number of available CPUs.
    -h, --help                  Show help.
    --version                   Show the version.
";

#[derive(Deserialize)]
struct Args {
    cmd_log: bool,
    cmd_log10: bool,
    cmd_nn_bound: bool,
    cmd_frequentist: bool,
    flag_knn: Option<usize>,
    flag_verbose: Option<String>,
    flag_delta: Option<f64>,
    flag_qstop: Option<usize>,
    flag_max_k: Option<usize>,
    flag_absolute: bool,
    flag_scale: bool,
    flag_nprocs: Option<usize>,
    arg_train: String,
    arg_test: String,
}

/// Returns `n` if `n` is odd, otherwise `n+1`.
fn next_odd(n: usize) -> usize {
    match n % 2 {
        0 => n + 1,
        _ => n,
    }
}

/// Computes the NN bound derived from Cover&Hart, given
/// the error and the number of labels.
fn nn_bound(error: f64, nlabels: usize) -> f64 {
    let nl = nlabels as f64;
    // Computing: (L-1)/L * (1 - (1 - L/(L-1)*error).sqrt())
    // with error = min(error, rg).
    let rg = (nl-1.)/nl;
    match error {
        e if e < rg => rg * (1. - (1. - nl/(nl-1.)*error).sqrt()),
        _ => rg,
    }
}

/// Returns a (boxed) closure determining how to compute k
/// given the number of training examples n.
fn k_from_n(args: &Args) -> Box<Fn(usize) -> usize> {
    if let Some(k) = args.flag_knn {
        Box::new(move |_| k)
    } else if args.cmd_nn_bound {
        Box::new(|_| 1)
    } else if args.cmd_log {
        Box::new(|n| next_odd(if n != 0 {
                                (n as f64).ln().ceil() as usize
                              } else {
                                1
                              }))
    } else if args.cmd_log10 {
        Box::new(|n| next_odd(if n != 0 {
                                (n as f64).log10().ceil() as usize
                              } else {
                                1
                              }))
    } else if args.cmd_frequentist {
        Box::new(move |_| 0)
    } else {
        panic!("this shouldn't happen");
    }
}

/// Prints several security measures that can be derived from a Bayes risk
/// estimate and Random guessing error.
fn print_all_measures(bayes_risk_estimate: f64, random_guessing: f64) {
    println!("Multiplicative Leakage: {}",
             multiplicative_leakage(bayes_risk_estimate, random_guessing));
    println!("Additive Leakage: {}",
             additive_leakage(bayes_risk_estimate, random_guessing));
    println!("Bayes security measure: {}",
             bayes_security_measure(bayes_risk_estimate, random_guessing));
    println!("Min-entropy Leakage: {}",
             min_entropy_leakage(bayes_risk_estimate, random_guessing));
}

/// Estimates security measures with a forward strategy (i.e., with an
/// increasing number of examples).
fn run_forward_strategy(mut estimator: Estimator, compute_nn_bound: bool,
                        nlabels: usize, mut convergence_checker: Option<ForwardChecker>,
                        verbose: Option<String>, train_x: Array2<f64>,
                        train_y: Array1<Label>) -> (f64, f64) {
    // Init verbose log file.
    let mut logfile = if let Some(fname) = verbose {
        let mut logfile = File::create(&fname)
                               .expect("couldn't open file for verbose logging");
        writeln!(logfile, "n, error-count, estimate").expect("failed to write to verbose file");
        Some(logfile)
    } else {
        None
    };

    // We keep track both of the minimum and of the last estimate.
    let mut min_error = 1.0;
    let mut last_error = 1.0;

    for (n, (x, y)) in train_x.outer_iter().zip(train_y.iter()).enumerate() {
        // Compute error.
        last_error = match estimator.next(n, &x, *y) {
            Ok(error) => error,
            Err(_) => {
                // TODO: this error is specific to k-NN. If we add new
                // methods we may want to change this.
                panic!("Could not add more examples: maybe increase --max-k?");
            },
        };

        // Compute NN bound by Cover and Hart if needed.
        if compute_nn_bound {
            last_error = nn_bound(last_error, nlabels);
        }

        if min_error > last_error {
            min_error = last_error;
        }

        if let Some(ref mut logfile) = logfile {
            writeln!(logfile, "{}, {}, {}", n, estimator.error_count(),
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


fn main() {
    // Parse args from command line.
    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.version(Some(env!("CARGO_PKG_VERSION").to_string()))
                                           .deserialize())
                            .unwrap_or_else(|e| e.exit());

    // Number of processes.
    if let Some(nprocs) = args.flag_nprocs {
        rayon::ThreadPoolBuilder::new()
                                 .num_threads(nprocs)
                                 .build_global()
                                 .unwrap();
    }

    // Load data.
    let (mut train_x, train_y) = load_data::<f64>(&args.arg_train)
                                .expect("[!] failed to load training data");
    let (mut test_x, test_y) = load_data::<f64>(&args.arg_test)
                                .expect("[!] failed to load test data");

    // Remap labels so they are zero-based increasing numbers.
    let (train_y, mapping) = vectors_to_ids(train_y.view()
                                            .into_shape((train_y.len(), 1))
                                            .unwrap(), None);
    let train_nlabels = mapping.len();
    // Remap test labels according to the mapping used for training labels.
    let (test_y, mapping) = vectors_to_ids(test_y.view()
                                           .into_shape((test_y.len(), 1))
                                           .unwrap(), Some(mapping));
    // The test labels should all have appeared in the training data;
    // the reverse is not necessary. If new labels appear in test_y,
    // the mapping is extended, so we can assert that didn't happen
    // as follows.
    let nlabels = mapping.len();
    // NOTE (6/11/18): this assertion could be removed with an optional
    // command line flag; indeed, to my understanding, this won't cause
    // problems to the estimation. However, for the time being I'll keep
    // it as it is, which is the "safest" option.
    assert_eq!(nlabels, train_nlabels,
               "Test data contains labels unseen in training data.
                Each test label should appear in the training data; the converse is not necessary");

    // (delta, q)-convergence checker
    let convergence_checker = if args.flag_delta.is_none() && args.flag_qstop.is_none() {
        // By default, run all (i.e., don't stop for (delta, q)-convergence.
        None
    } else if let Some(delta) = args.flag_delta {
        let q = match args.flag_qstop {
            Some(q) => if q < train_x.len() { q } else { train_x.len()-1 },
            None => (train_x.len() as f64 * 0.1) as usize,
        };
        println!("will stop when (delta={}, q={})-converged", delta, q);
        Some(ForwardChecker::new(&vec![delta], q, !args.flag_absolute))
    } else {
        panic!("--qstop should only be specified with --delta");
    };

    // Scale features.
    if train_x.cols() > 1 && args.flag_scale {
        println!("scaling features");
        scale01(&mut train_x);
        scale01(&mut test_x);
    }

    // Init estimator.
    let estimator = if args.cmd_frequentist {
        Estimator::Frequentist(FrequentistEstimator::new(nlabels,
                                 &test_x.view(),
                                 &test_y.view()))
    } else {
        // How k is computed w.r.t. n.
        let kn = k_from_n(&args);
        // max_k specifies the maximum number of neighbors to store
        // (excluding ties); a smaller max_k improves performances, but
        // its value should be sufficiently large to give correct
        // results once we've seen all the training data.
        let max_k = match args.flag_max_k {
            Some(max_k) => max_k,
            None => kn(train_x.rows()),
        };

        Estimator::KNN(KNNEstimator::new(&test_x.view(), &test_y.view(),
                                         1, max_k), kn)
    };

    let random_guessing = estimate_random_guessing(&test_y.view());
    println!("Random guessing error: {}", random_guessing);
    println!("Estimating leakage measures...");

    let (min_error, last_error) = run_forward_strategy(estimator, args.cmd_nn_bound,
                                                       nlabels, convergence_checker,
                                                       args.flag_verbose, train_x,
                                                       train_y);

    println!();
    println!("Final estimate: {}", last_error);
    print_all_measures(last_error, random_guessing);
    println!();

    println!("Minimum estimate: {}", min_error);
    print_all_measures(min_error, random_guessing);
}
