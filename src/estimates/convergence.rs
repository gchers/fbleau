//! This module provides tools for verifying whether an estimate
//! has converged.
//!
//! It implements convergence heuristics based on relative/absolute
//! convergence.
use std::collections::{HashMap, VecDeque};
use ordered_float::OrderedFloat;

use crate::estimates::{some_or_error};

/// Returns relative or absolute change between two measurements.
fn change(a: f64, b: f64, relative: bool) -> f64 {
    if relative {
        (a - b).abs() / b
    } else {
        (a - b).abs()
    }
}

/// `ForwardChecker` should be used for checking convergence of
/// estimates in a "forward" direction (i.e., when one training example
/// is _added_ each time).
///
/// It allows checking for relative or absolute convergence:
/// we declare convergence if an estimate did not change (in relative
/// or absolute sense) more than some `delta` for at least `q` steps.
/// `ForwardChecker` allows measuring `delta`-convergence for several
/// values of `delta`.
pub struct ForwardChecker {
    // A double-ended queue keeping track of all the estimates for which
    // next_delta-convergence happens.
    estimates: VecDeque<f64>,
    // The index (in the original training data) of the estimate
    // that corresponds to estimates[0].
    first_n: usize,
    // A hash map where delta_converged[delta], for some delta,
    // is either None or the index in the original data for which the
    // estimate started converging.
    delta_converged: HashMap<OrderedFloat<f64>, Option<usize>>,
    // Keeps track of the next deltas (if any) for which we
    // seek convergence. Sorted in ascending order.
    next_deltas: Vec<f64>,
    // We declare convergence when there are at least q estimates
    // whose absolute/relative change is < delta, for some delta.
    q: usize,
    // Use relative or absolute change.
    relative: bool,
}

impl ForwardChecker {
    pub fn new(deltas: &[f64], q: usize, relative: bool) -> ForwardChecker {
        assert!(!deltas.is_empty());

        // Deltas need to be sorted.
        let mut deltas = deltas.to_owned();
        deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(*deltas.get(0).unwrap() > 0.);

        ForwardChecker {
            estimates: VecDeque::new(),
            first_n: 0,
            delta_converged: deltas.iter()
                                    .map(|d| (OrderedFloat::from(*d), None))
                                    .collect(),
            next_deltas: deltas,
            q,
            relative,
        }
    }

    pub fn get_converged(&self) -> HashMap<OrderedFloat<f64>, Option<usize>> {
        self.delta_converged.clone()
    }

    pub fn get_not_converged(&self) -> HashMap<OrderedFloat<f64>, Option<usize>> {
        let mut converged = HashMap::new();
        let last_e = match self.estimates.back() {
            Some(est) => est,
            None => return converged,
        };

        let mut next_deltas = self.next_deltas.clone();
        let mut delta = match next_deltas.pop() {
            Some(delta) => delta,
            None => return converged,
        };

        'outer: for (i, est) in self.estimates.iter().enumerate() {
            while change(*est, *last_e, self.relative) < delta {
                converged.insert(OrderedFloat::from(delta), Some(self.first_n + i));
                delta = match next_deltas.pop() {
                    Some(delta) => delta,
                    None => break 'outer,
                };
            }
        }
        converged
    }

    pub fn all_converged(&self) -> bool {
        self.next_deltas.is_empty()
    }

    pub fn add_estimate(&mut self, e: f64) {
        if self.all_converged() {
            return;
        }
        self.estimates.push_back(e);
        // We may have converged for more than one delta, so we need
        // to try until no more updates occour.
        while self.update_convergence() {};
    }

    pub fn get_last_change(&self) -> Result<f64, ()> {
        let first_e = some_or_error(self.estimates.front())?;
        let last_e = some_or_error(self.estimates.back())?;

        Ok(change(*first_e, *last_e, self.relative))
    }

    fn update_convergence(&mut self) -> bool {
        let delta = match self.next_deltas.last() {
            Some(delta) => *delta,
            // Everything already converged.
            None => return false,
        };
        // Get last estimate value.
        let last_e = match self.estimates.back() {
            Some(est) => *est,
            None => return false,
        };
        // Remove from the front all the estimates for which
        // delta-convergence doesn't hold.
        // NOTE: the unwrap() here is justified by the fact that
        // self.estimates contains at least `last_e` itself, and
        // (e - e) = 0 < delta for any valid delta.
        while change(*self.estimates.front().unwrap(), last_e, self.relative) >= delta {
            let _ = self.estimates.pop_front();
            self.first_n += 1;
        }
        // Check if we converged.
        if self.estimates.len() >= self.q {
            let c = self.delta_converged.get_mut(&OrderedFloat::from(delta)).unwrap();
            *c = Some(self.first_n);
            // Update next delta.
            self.next_deltas.pop();
            if self.next_deltas.is_empty() {
                // We don't need estimates anymore.
                self.estimates.clear();
            }
            return true;
        }
        false
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_checker_init() {
        let checker = ForwardChecker::new(&vec![0.1, 0.02, 0.4, 0.001], 1000,
                                          false);

        assert_eq!(checker.next_deltas, vec![0.001, 0.02, 0.1, 0.4]);
    }

    #[test]
    fn forward_checker_convergence() {
        let deltas = vec![0.11, 0.06, 0.02, 0.02, 0.019, 0.002, 0.0001];
        let mut checker = ForwardChecker::new(&deltas, 4, false);
        
        let estimates = vec![0.9, 0.9, 0.8, 0.7, 0.7, 0.7, 0.65, 0.64, 0.63,
                             0.63, 0.62, 0.62, 0.6, 0.6, 0.6, 0.599];
        let expected_conv: Vec<Option<usize>> = vec![Some(2), Some(3), Some(8),
                                                     Some(8), Some(8), 
                                                     Some(12), None];

        for e in estimates {
            checker.add_estimate(e);
        }
        //assert!(checker.all_converged());
        let converged = checker.get_converged();
        for (delta, n) in deltas.iter().zip(expected_conv) {
            assert_eq!(converged.get(&OrderedFloat::from(*delta)).unwrap(), &n);
        }
    }

    #[test]
    fn absolute_convergence() {
        let deltas = vec![0.15, 0.1, 0.02, 0.02, 0.019, 0.002, 0.0017];
        let estimates = vec![0.9, 0.9, 0.8, 0.7, 0.7, 0.7, 0.65, 0.64, 0.63,
                             0.63, 0.62, 0.62, 0.6, 0.6, 0.6, 0.599];

        let mut fwchecker = ForwardChecker::new(&deltas, 4, false);
        
        let expected_conv: Vec<Option<usize>> = vec![Some(2), Some(3), Some(8),
                                                     Some(8), Some(8), 
                                                     Some(12), Some(12)];

        for e in estimates {
            fwchecker.add_estimate(e);
        }

        assert!(fwchecker.all_converged());

        let fwconverged = fwchecker.get_converged();

        for (delta, n) in deltas.iter().zip(expected_conv) {
            assert_eq!(fwconverged.get(&OrderedFloat::from(*delta)).unwrap(), &n);
        }
    }
}
