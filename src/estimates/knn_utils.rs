use ndarray::prelude::*;
use strsim::generic_levenshtein;


/// Computes the NN bound derived from Cover&Hart, given
/// the error and the number of labels.
pub fn nn_bound(error: f64, nlabels: usize) -> f64 {
    let nl = nlabels as f64;
    // Computing: (L-1)/L * (1 - (1 - L/(L-1)*error).sqrt())
    // with error = min(error, rg).
    let rg = (nl-1.)/nl;
    match error {
        e if e < rg => rg * (1. - (1. - nl/(nl-1.)*error).sqrt()),
        _ => rg,
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

/// Strategies for selecting `k` for k-NN given the number of
/// training examples `n`.
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KNNStrategy {
    Ln,
    Log10,
    // We do not want to parse the following ones from the command line.
    // So we ask serde to skip them.
    #[serde(skip)]
    NN,
    #[serde(skip)]
    FixedK(usize),
    #[serde(skip)]
    Custom(Box<dyn Fn(usize) -> usize>),
}

pub fn knn_strategy(strategy: KNNStrategy) -> Box<dyn Fn(usize) -> usize> {
    match strategy {
        KNNStrategy::NN => Box::new(move |_| 1),
        KNNStrategy::FixedK(k) => Box::new(move |_| k),
        KNNStrategy::Ln => Box::new(move |n|
                                    next_odd(if n != 0 {
                                                (n as f64).ln().ceil() as usize
                                             } else { 1 })),
        KNNStrategy::Log10 => Box::new(move |n|
                                    next_odd(if n != 0 {
                                                (n as f64).log10().ceil() as usize
                                             } else { 1 })),
        KNNStrategy::Custom(custom) => custom,
    }
}

/// Returns `n` if `n` is odd, otherwise `n+1`.
fn next_odd(n: usize) -> usize {
    match n % 2 {
        0 => n + 1,
        _ => n,
    }
}
