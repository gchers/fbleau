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
//!     fbleau <estimate> [--knn-strategy=<strategy>] [options] <train> <eval>
//!
//! ## Estimates
//!
//! Currently available estimates:
//!
//! - nn
//! - knn
//! - frequentist
//! - nn-bound
//!
//! NOTE: The `frequentist` and `nn` strategies only converge if the
//! observation space is finite. The `knn` estimator is guaranteed to
//! converge (given enough data) even if the observation space is continuous.
//! The `nn-bound` works for both continuous/finite spaces, but it guarantees
//! to be a lower bound of the Bayes risk.
//!
//! The `knn` option must be accompained by a `--knn-strategy` flag, whose
//! value is in:
//! - ln
//! - log10
//! The choice between the two cannot be done a priori: one should try both,
//! and see which one produces the smallest estimate.
extern crate ndarray;
extern crate docopt;
#[macro_use]
extern crate serde;
extern crate itertools;
extern crate strsim;

extern crate fbleau;

use docopt::Docopt;

use fbleau::estimates::*;
use fbleau::security_measures::*;
use fbleau::fbleau_estimation::run_fbleau;
use fbleau::utils::load_data;

const USAGE: &str = "
Estimate k-NN error and convergence.

Usage: fbleau <estimate> [--knn-strategy=<strategy>] [options] <train> <eval>
       fbleau (--help | --version)

Arguments:
    estimate:   nn              Nearest Neighbor. Converges only if the
                                observation space is finite.
                knn             k-NN rule. Converges for finite/continuous
                                observation spaces.
                frequentist     Frequentist estimator. Converges only if the
                                observation space is finite.
    knn-strategy: ln            k-NN with k = ln(n).
                  log10         k-NN with k = log10(n).
    train                       Training data (.csv file).
    eval                        Evaluation data (.csv file).

Options:
    --logfile=<fname>           Log estimates at each step.
    --delta=<d>                 Delta for delta covergence.
    --qstop=<q>                 Number of examples to declare
                                delta-convergence. Default is 10% of
                                training data.
    --absolute                  Use absolute convergence instead of relative
                                convergence.
    --scale                     Scale features before running k-NN
                                (only makes sense for objects of 2 or more
                                dimensions).
    --distance=<name>           Distance metric in (\"euclidean\",
                                \"levenshtein\").
    -h, --help                  Show help.
    --version                   Show the version.
";

#[derive(Deserialize)]
struct Args {
    arg_estimate: Estimate,
    flag_knn_strategy: Option<KNNStrategy>,
    flag_logfile: Option<String>,
    flag_delta: Option<f64>,
    flag_qstop: Option<usize>,
    flag_absolute: bool,
    flag_scale: bool,
    flag_distance: Option<String>,
    arg_train: String,
    arg_eval: String,
}


fn main() {
    // Parse args from command line.
    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.version(Some(env!("CARGO_PKG_VERSION")
                                                            .to_string()))
                                           .deserialize())
                            .unwrap_or_else(|e| e.exit());

    // Load data.
    let (train_x, train_y) = load_data::<f64>(&args.arg_train)
                                .expect("[!] failed to load training data");
    let (eval_x, eval_y) = load_data::<f64>(&args.arg_eval)
                                .expect("[!] failed to load evaluation data");

    let (min_error, _, random_guessing) = 
        run_fbleau(train_x, train_y, eval_x, eval_y, args.arg_estimate,
                   args.flag_knn_strategy, args.flag_distance, args.flag_logfile,
                   args.flag_delta, args.flag_qstop, args.flag_absolute,
                   args.flag_scale);

    println!();
    println!("Minimum estimate: {}", min_error);
    print_all_measures(min_error, random_guessing);
}
