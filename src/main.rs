extern crate ndarray;
extern crate docopt;
#[macro_use]
extern crate serde_derive;
extern crate itertools;

extern crate fbleau;

mod utils;

use ndarray::*;
use docopt::Docopt;

use fbleau::Label;
use fbleau::estimates::*;
use utils::{load_data, vectors_to_ids, scale01};


const USAGE: &'static str = "
Estimate k-NN error and convergence.

Usage: fbleau log [options] <train> <test>
       fbleau log10 [options] <train> <test>
       fbleau nn-bound [options] <train> <test>
       fbleau --knn=<k> [options] <train> <test>
       fbleau frequentist [options] <train> <test>
       fbleau (--help | --version)

Options:
    --delta=<d>                 Delta for delta covergence [default: 0.1].
                                Multiple deltas can be specified as
                                comma-separated values.
    --qstop=<q>                 Number of examples to declare
                                delta-convergence. Default is 10% of
                                training data.
    --run-all                   Don't stop running after convergence.
    --abs                       Use absolute convergence instead of relative
                                convergence.
    --max-k=<k>                 Number of neighbors to store, initially,
                                for each test point [default: 100].
    --no-scale                  Don't scale features before running k-NN
                                (only makes sense for objects of 2 or more
                                dimensions).
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
    flag_delta: String,
    flag_qstop: Option<usize>,
    flag_max_k: usize,
    flag_abs: bool,
    flag_no_scale: bool,
    flag_run_all: bool,
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

/// Parses a string of deltas specified as comma-separated values.
fn parse_deltas(deltas: &str) -> Vec<f64> {
    deltas.split(",")
        .map(|s| s.parse::<f64>().expect("couldn't parse deltas"))
        .collect::<Vec<_>>()
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

fn run_forward_strategy(args: &Args, nlabels: usize, deltas: &Vec<f64>, q: usize,
                        max_k: usize, kn: Box<Fn(usize) -> usize>,
                        mut train_x: Array2<f64>, train_y: Array1<Label>,
                        mut test_x: Array2<f64>, test_y: Array1<Label>) {

    // Initialize.
    let mut estimator = if args.cmd_frequentist {
        // FIXME: this WILL lose precision, so we need to work with
        // f64 that are actually integers in practice.
        let train_x = train_x.map(|x| *x as usize);
        let test_x = test_x.map(|x| *x as usize);

        // NOTE: we remap even if feature vectors have size 1.
        // FIXME: there's a warning on train_ids not being used;
        // to fix that, we could simply remove the next line, and record
        // "mapping" from the line below. However, I want to be certain
        // the result is identical, and I don't have time to test this now.
        let (train_ids, mapping) = vectors_to_ids(train_x.view(), None);
        let (test_ids, _) = vectors_to_ids(test_x.view(), Some(mapping.clone()));

        Estimator::Frequentist(FrequentistEstimator::new(nlabels,
                                 &test_ids.view(),
                                 &test_y.view()), mapping)
    } else {
        if train_x.cols() > 1 && !args.flag_no_scale {
            println!("scaling features");
            scale01(&mut train_x);
            scale01(&mut test_x);
        }
        Estimator::KNN(KNNEstimator::new(&test_x.view(), &test_y.view(),
                                         1, max_k), kn)
    };


    let mut convergence_checker = ForwardChecker::new(deltas, q, !args.flag_abs);
    // NOTE: temporary fix.
    let kn = k_from_n(&args);

    // Print header.
    println!("n, k, error-count, error, bound");
    let mut min_error = 1.0;

    for (n, (x, y)) in train_x.outer_iter().zip(train_y.iter()).enumerate() {

        // Compute error.
        let error = match estimator.next(n, &x, *y) {
            Ok(error) => error,
            Err(_) => {
                // FIXME: we should exit at the end of the loop.
                println!("stopped because could not remove any more examples");
                break;
            },
        };

        if min_error > error {
            min_error = error;
        }

        // Compute NN bound by Cover and Hart if requested.
        let bound = if args.cmd_nn_bound {
            nn_bound(error, nlabels)
        } else { -1. };

        let k = kn(n);
        println!("{}, {}, {}, {}, {}", n, k, estimator.error_count(), error, bound);

        // Check convergence.
        if args.cmd_nn_bound {
            convergence_checker.add_estimate(bound);
        }
        else {
            convergence_checker.add_estimate(error);
        }

        if !args.flag_run_all && convergence_checker.all_converged() {
            break;
        }
    }

    // Print for what deltas we converged after how many examples.
    for (delta, converged) in convergence_checker.get_converged() {
        if let Some(converged) = converged {
            println!("[*] {}-convergence after {} examples", delta, converged);
        }
    }
    for (delta, converged) in convergence_checker.get_not_converged() {
        if let Some(converged) = converged {
            println!("[*] {}-convergence (< q!) after {} examples", delta, converged);
        }
    }

    println!("Minimum estimate: {}", min_error);
}


fn main() {
    // Parse args from command line.
    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.deserialize())
                            .unwrap_or_else(|e| e.exit());

    let (train_x, train_y) = load_data::<f64>(&args.arg_train)
                                .expect("[!] failed to load training data");
    let (test_x, test_y) = load_data::<f64>(&args.arg_test)
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

    let max_k = args.flag_max_k;

    // Deltas and q for delta-convergence.
    let deltas = parse_deltas(&args.flag_delta);
    let q = match args.flag_qstop {
        Some(q) => if q < train_x.len() { q } else { train_x.len()-1 },
        None => (train_x.len() as f64 * 0.1) as usize,
    };
    println!("Convergence q: {}", q);

    // How k is computed w.r.t. n.
    let kn = k_from_n(&args);

    run_forward_strategy(&args, nlabels, &deltas, q, max_k, kn,
                         train_x, train_y, test_x, test_y);
}
