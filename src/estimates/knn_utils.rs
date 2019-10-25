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

/// Stragegies for selecting `k` for k-NN given the number of
/// training examples `n`.
pub enum KNNStrategy {
    NN,
    FixedK(usize),
    Ln,
    Log10,
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
