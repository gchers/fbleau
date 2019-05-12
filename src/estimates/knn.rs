//! Fast k-NN error estimates for discrete and continuous
//! output space.
//!
//! This allows estimating the error of a k-NN classifier (with possibly
//! changing k) on a test set, given training data.
//! This module only exposes one public structure: `KNNEstimator`, which can be
//! used as follows:
//! 1) "train" `KNNEstimator` on the full training data, with some test data;
//! 2) get the error of k-NN on test data;
//! 3) remove one training example with `remove_one()`;
//! 4) repeat from 2), until training data is finished.
//!
//! # Examples
//!
//! ```
//! #[macro_use(array)]
//! extern crate ndarray;
//! extern crate fbleau;
//!
//! # fn main() {
//! use ndarray::*;
//! use fbleau::estimates::*;
//!
//! let train_x = array![[8.],
//!                      [7.],
//!                      [6.],
//!                      [5.],
//!                      [4.]];
//! let train_y = array![0, 0, 0, 1, 0];
//! let test_x = array![[3.],
//!                     [0.],
//!                     [6.],
//!                     [1.],
//!                     [6.],
//!                     [4.],
//!                     [5.]];
//! let test_y = array![0, 0, 2, 1, 0, 1, 0];
//! let max_k = 8;
//! 
//! let k = 3;
//! let mut knn = KNNEstimator::from_data(&train_x.view(), &train_y.view(),
//!                             &test_x.view(), &test_y.view(), k, max_k);
//!
//! assert_eq!(knn.get_error(), 0.42857142857142855);
//!
//! knn.add_example(&array![3.].view(), 1);
//! assert_eq!(knn.get_error(), 0.42857142857142855);
//!
//! // Change k.
//! knn.set_k(5);
//! knn.add_example(&array![2.].view(), 1);
//! assert_eq!(knn.get_error(), 0.42857142857142855);
//!
//! knn.add_example(&array![1.].view(), 2);
//! assert_eq!(knn.get_error(), 0.42857142857142855);
//! # }
//! ```
use std;
use ndarray::*;
use ndarray_parallel::rayon::prelude::*;
use std::collections::HashMap;
use ndarray_parallel::prelude::*;
use ordered_float::OrderedFloat;
use std::cmp::Ordering;

use Label;

/// Nearest neighbors to a test object.
#[derive(Debug)]
struct Neighbor {
    // Distance from the object of which this is a neighbor.
    distance: f64,
    // Class of this object.
    label: Label,
}

impl Neighbor {
    /// Constructs a new Neighbor.
    fn new(distance: f64, label: Label) -> Neighbor {
        Neighbor {
            distance: distance,
            label: label,
        }
    }
}

// Ordering for Neighbor.
impl Ord for Neighbor {
    fn cmp(&self, other: &Neighbor) -> Ordering {
        let self_d = OrderedFloat::from(self.distance);
        let other_d = OrderedFloat::from(other.distance);

        self_d.cmp(&other_d)
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Neighbor) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Neighbor) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}


/// Contains the nearest neighbors of some test object x.
#[derive(Debug)]
struct NearestNeighbors<F>
where F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64 {
    // Test object x.
    x: Array1<f64>,
    // List of neighbors, sorted in increasing order by their distance
    // from x.
    neighbors: Vec<Neighbor>,
    // Neighbors that have the same distance to x as neighbors[-1].
    // Specifically, extra_ties[y], is the number of neighbors with the same
    // distance (i.e., extra_ties_dist) as neighbors[-1] that have label y.
    extra_ties: HashMap<Label, usize>,
    // Distance of ties. None if there are no ties.
    extra_ties_dist: Option<f64>,
    // Smallest index in `neighbors` which was last updated.
    // This means that if we want a prediction for k > updated_k,
    // we need to update the previous prediction, otherwise, we can re-use
    // the previous one. Note that, to handle extra_ties, updated_k actually
    // points to the beginning of them; so, for example, if the
    // neighbors' distances are [0, 1, 1, 2, 2, 2, 2], and the last updated
    // index was 5 (distance 2), then updated_k = 3 (the first "2").
    updated_k: usize,
    // Maximum number of neighbors (excluding extra_ties).
    max_k: usize,
    distance: F,
}

impl<F> NearestNeighbors<F>
where F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64 + Copy {
    /// Init a list of neighbors for a specified test object x.
    fn new(x: &ArrayView1<f64>, max_k: usize, distance: F) -> NearestNeighbors<F> {
        NearestNeighbors {
            x: x.to_owned(),
            // Capacity: max_k + 1 for when we insert a new element and then
            // remove another one from the tail.
            neighbors: Vec::with_capacity(max_k + 1),
            extra_ties: HashMap::new(),
            extra_ties_dist: None,
            updated_k: 0,
            max_k: max_k,
            distance: distance,
        }
    }

    /// Init a list of neighbors for a specified test object x, and given
    /// training data.
    ///
    /// # Arguments
    /// * `x` - Test object.
    /// * `train_x` - Training objects.
    /// * `train_y` - Training labels.
    /// * `max_k` - Maximum number of neighbors to store for test object x.
    ///
    fn from_data(x: &ArrayView1<f64>, train_x: &ArrayView2<f64>,
                 train_y: &ArrayView1<Label>, max_k: usize, distance: F) -> NearestNeighbors<F> {
        assert!(max_k > 0);

        let mut knn = NearestNeighbors::new(x, max_k, distance);

        for (xi, yi) in train_x.outer_iter().zip(train_y) {
            // NOTE: we use std::usize::MAX as a bogus label to split ties in
            // predict(); if this becomes a problem, predict() needs to be
            // fixed as well.
            assert!(*yi != std::usize::MAX,
                    "label {} is too large and currently not supported", *yi);

            knn.add_example(&xi, *yi, distance);
        }

        knn
    }

    /// Predict label according to k-NN rule, for specified k, without
    /// accounting for ties.
    fn predict_no_ties(&self, k: usize) -> Result<Label, ()> {
        if k > self.neighbors.len() {
            return Err(());
        }

        let mut label_count = HashMap::new();

        let mut y_pred = 0;
        let mut y_count = 0;

        for neigh in self.neighbors.iter().take(k) {
            let count = label_count.entry(neigh.label).or_insert(0);
            *count += 1;
            if *count > y_count {
                y_pred = neigh.label;
                y_count = *count;
            }
        }
        Ok(y_pred)
    }

    /// Predict label according to k-NN rule, for specified k.
    fn predict(&self, k: usize) -> Result<Label, ()> {
        if k > self.neighbors.len() {
            return Err(());
        }

        // TODO: remember prediction counts for some k,
        // and update them when add_example() is called?

        // If the k-th element does not exist or has distance larger
        // than the (k-1)-th element then we don't split ties.
        let ties_d = self.neighbors[k-1].distance;
        let no_ties = (self.neighbors.len() <= k) || (self.neighbors[k].distance > ties_d);
        // Do self.extra_ties play a part in this prediction?
        let extra_ties_matter = (self.extra_ties.len() > 0) && (k == self.neighbors.len());

        if no_ties && !extra_ties_matter {
            return self.predict_no_ties(k);
        }

        // Behold!
        // Follows a couple of hours worth of code craziness which makes
        // our ties splitting code run in O(max_k) (in fact, it's usually going
        // to run in O(k+k_ties), where k_ties is the number of ties,
        // and generally k+k_ties << max_k).
        // We keep two hash maps: one accounting for labels' count (with
        // a placeholder symbol std::usize::MAX when a label is part of ties),
        // and the other one counting labels in ties.
        let mut label_count = HashMap::new();
        let mut ties_label_count = HashMap::new();

        let mut y_pred = 0;
        let mut y_count = 0;
        let mut ties_y_pred = 0;
        let mut ties_y_count = 0;
        // "Symbolic" value for ties label.
        const TIES_LABEL: usize = std::usize::MAX;

        for (i, neigh) in self.neighbors.iter().enumerate() {
            if neigh.distance != ties_d {
                if i >= k {
                    break;
                }
                let count = label_count.entry(neigh.label).or_insert(0);
                *count += 1;
                if *count > y_count {
                    y_pred = neigh.label;
                    y_count = *count;
                }
            }
            else {
                // Count labels within ties.
                let count = ties_label_count.entry(neigh.label).or_insert(0);
                *count += 1;
                if *count > ties_y_count {
                    ties_y_pred = neigh.label;
                    ties_y_count = *count;
                }
                // We have a special "symbol" for counting
                // how many elements with ties exist: ties_y.
                // However, we only count them up to k.
                let count = label_count.entry(TIES_LABEL).or_insert(0);
                if i >= k {
                    continue;
                }
                *count += 1;
                if *count > y_count {
                    y_pred = TIES_LABEL;
                    y_count = *count;
                }
            }
        }

        // Include self.extra_ties, if extra ties have the same distance
        // as ties_d.
        if Some(ties_d) == self.extra_ties_dist {
            for (y, extra_count) in &self.extra_ties {
                let count = ties_label_count.entry(*y).or_insert(0);
                *count += extra_count;
                if *count > ties_y_count {
                    ties_y_pred = *y;
                    ties_y_count = *count;
                }
            }
        }

        // If the predicted label is the same as the TIES_LABEL placeholder,
        // we can output the best label among ties. Otherwise, we need to
        // count which is the most frequent overall.
        if y_pred == TIES_LABEL {
            y_pred = ties_y_pred;
        }
        else {
            let mut count = *label_count.get(&TIES_LABEL)
                                    .expect("[!] unexpected error in splitting ties");
            if let Some(c) = label_count.get(&ties_y_pred) {
                count += c;
            }

            if count > y_count {
                y_pred = ties_y_pred;
            }
        }

        Ok(y_pred)
    }

    /// Returns the index of the first neighbor with the same distance as
    /// self.neighbors[i].
    fn first_of_ties(&self, mut i: usize) -> usize {
        if i == 0  || self.neighbors.len() == 0 {
            return 0;
        }

        let d = self.neighbors.get(i)
                              .expect("first_of_ties() called on wrong index")
                              .distance;

        while let Some(neigh) = self.neighbors.get(i-1) {
            if neigh.distance != d {
                break;
            }
            i -= 1;
            if i == 0 {
                break;
            }
        }
        i
    }

    /// Adds a new example.
    fn add_example(&mut self, x: &ArrayView1<f64>, y: Label, distance: F) -> bool {
        let d = distance(x, &self.x.view());

        if self.neighbors.len() < self.max_k {
            // If still filling, insert sorted.
            let new = Neighbor::new(d, y);
            let pos = self.neighbors.binary_search(&new).unwrap_or_else(|e| e);
            self.neighbors.insert(pos, new);

            // Update updated_k with ties.
            self.updated_k = self.first_of_ties(pos);

        }
        else if self.neighbors.last().unwrap().distance < d {
            return false;
        }
        else if self.neighbors.last().unwrap().distance == d {
            // Handle ties.
            if self.extra_ties.len() == 0 {
                self.extra_ties_dist = Some(d);
            }
            // Could do this after.
            {
                let count = self.extra_ties.entry(y).or_insert(0);
                *count += 1;
            }

            // Update updated_k with ties.
            self.updated_k = self.first_of_ties(self.neighbors.len()-1);
        }
        else {
            // Insert sorted.
            let new = Neighbor::new(d, y);
            let pos = self.neighbors.binary_search(&new).unwrap_or_else(|e| e);
            self.neighbors.insert(pos, new);

            // Update updated_k with ties.
            self.updated_k = self.first_of_ties(pos);

            if let Some(removed) = self.neighbors.pop() {
                let last_neigh = self.neighbors.last().unwrap();
                // Either add to ties, or remove all ties.
                if last_neigh.distance == removed.distance {
                    //self.ties.push(removed);
                    if self.extra_ties.len() == 0 {
                        self.extra_ties_dist = Some(removed.distance);
                    }
                    // TODO: could remove this check if we're sure it is
                    // updated elsewhere correctly.
                    else {
                        assert_eq!(Some(removed.distance), self.extra_ties_dist);
                    }
                    let count = self.extra_ties.entry(removed.label).or_insert(0);
                    *count += 1;
                }
                else {
                    self.extra_ties.clear();
                    self.extra_ties_dist = None;
                }
            }
        }
        true
    }
}

/// Keeps track of the error of a k-NN classifier, with the possibility
/// of changing k and removing training examples.
#[derive(Debug)]
pub struct KNNEstimator<F>
where F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64 {
    // max_k nearest neighbors for each test object.
    neighbors: Vec<NearestNeighbors<F>>,
    // Error for each test object.
    pub errors: Vec<f64>,
    // Current prediction for each test label.
    pub predictions: Vec<Label>,
    // True test labels.
    labels: Vec<Label>,
    // Last queried k.
    current_k: usize,
    // k-NN count, for k = current_k.
    pub k_error_count: f64,
    // Size of training data. The next training example to be removed by
    // remove_one() is n-1. The next training example to be added by
    // add_example() is n-1.
    n: usize,
}

impl<F> KNNEstimator<F>
where F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64 + Send + Sync + Copy {
    /// Create a new k-NN estimator.
    pub fn new(test_x: &ArrayView2<f64>, test_y: &ArrayView1<Label>,
           k: usize, max_k: usize, distance: F) -> KNNEstimator<F> {
        assert_eq!(test_x.rows(), test_y.len());
        assert!(test_y.len() > 0);

        let neighbors = test_x.outer_iter()
                              .into_par_iter()
                              .map(|x| NearestNeighbors::new(&x, max_k, distance))
                              .collect::<Vec<_>>();
        // We initially set all predictions to 0. Therefore, we need to
        // adjust the error count accordingly. Note that this is updated as
        // soon as add_example() is called.
        // A more proper way to do this in Rust would be to set
        // predictions (and errors, k_error_count, ...) to an Option value.
        // TODO: should we?
        let errors = test_y.iter()
                           .map(|y| if *y != 0 { 1. } else { 0. })
                           .collect::<Vec<_>>();
        let error_count = errors.iter().sum();
        KNNEstimator {
            neighbors: neighbors,
            errors: errors,
            predictions: vec![0; test_y.len()],
            labels: test_y.to_vec(),
            current_k: k,
            k_error_count: error_count,
            n: 0,
        }
    }

    /// Create a k-NN estimator from training and test set.
    pub fn from_data(train_x: &ArrayView2<f64>, train_y: &ArrayView1<Label>,
           test_x: &ArrayView2<f64>, test_y: &ArrayView1<Label>,
           k: usize, max_k: usize, distance: F) -> KNNEstimator<F> {
        assert_eq!(train_x.cols(), test_x.cols());
        assert_eq!(train_x.rows(), train_y.len());
        assert_eq!(test_x.rows(), test_y.len());
        assert!(train_x.len() > 0);
        assert!(test_x.len() > 0);

        let neighbors = test_x.outer_iter()
                            .into_par_iter()
                            .map(|x| NearestNeighbors::from_data(&x,
                                                           &train_x.view(),
                                                           &train_y.view(),
                                                           max_k,
                                                           distance))
                            .collect::<Vec<_>>();

        let mut knn_error = 0.;
        let mut errors = Vec::with_capacity(test_y.len());
        let mut predictions = Vec::with_capacity(test_y.len());

        for (neigh, y) in neighbors.iter().zip(test_y) {
            let pred = neigh.predict(k)
                            .expect("unexpected error");
            let error = if pred != *y { 1. } else { 0. };

            predictions.push(pred);
            errors.push(error);
            knn_error += error;
        }

        KNNEstimator {
            neighbors: neighbors,
            errors: errors,
            predictions: predictions,
            labels: test_y.to_vec(),
            current_k: k,
            k_error_count: knn_error,
            n: train_y.len(),
        }
    }

    /// Update all predictions and errors.
    ///
    /// Called when when k changes.
    fn update_all(&mut self) -> Result<(), ()> {

        for (neigh, y, p, e) in izip!(&self.neighbors, &self.labels,
                                      &mut self.predictions, &mut self.errors) {
            let pred = neigh.predict(self.current_k)?;
            if pred == *p {
                continue;
            }
            let error = if pred != *y { 1. } else { 0. };

            self.k_error_count += error - *e;
            *p = pred;
            *e = error;
        }
        Ok(())
    }

    pub fn add_example(&mut self, x: &ArrayView1<f64>, y: Label, distance: F) -> Result<(), ()> {
        // We copy because we're using them in the closure below.
        let current_k = self.current_k;
        self.n += 1;    // NOTE: need to update here, before possible errors.

        // We do all this in this gigantic closure so that we can
        // parallelize it with rayon.
        let error_updates: Result<Vec<_>, ()> = self.neighbors.par_iter_mut()
            .zip(&self.labels)
            .zip(&mut self.predictions)
            .zip(&mut self.errors)
            // The closure in the following filter_map() returns an
            // Option<Result<f64, ()>>, where None is returned if there's
            // no need to update, Some(Ok(update)) is returned when the
            // current error needs updating +=update, and Some(Err(()))
            // is returned when an error occours;
            // I could not find a better way to catch and return the
            // errors within iter(), but maybe there's a better way.
            .filter_map(|(((neigh, true_y), old_pred), old_error)| {
                if neigh.add_example(x, y, distance) {
                    if neigh.updated_k > current_k {
                        return None;
                    }

                    // Update prediction.
                    let pred = match neigh.predict(current_k) {
                        Ok(pred) => pred,
                        _ => return Some(Err(())),
                    };
                    if pred == *old_pred {
                        return None;
                    }

                    // Update error.
                    let error = if pred != *true_y { 1. } else { 0. };

                    let update = error - *old_error;
                    *old_error = error;
                    *old_pred = pred;
                    return Some(Ok(update));
                }
                None
            })
            .collect();

        self.k_error_count += error_updates?.iter().sum::<f64>();

        Ok(())
    }

    /// Returns the error for the current k.
    pub fn get_error(&self) -> f64 {
        self.k_error_count / self.labels.len() as f64
    }
    
    /// Returns the error count for the current k.
    pub fn get_error_count(&self) -> usize {
        self.k_error_count as usize
    }

    /// Changes the k for which k-NN predictions are given.
    pub fn set_k(&mut self, k: usize) -> Result<(), ()> {
        if k != self.current_k {
            self.current_k = k;
            self.update_all()?;
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use estimates::*;

    #[test]
    fn knn_init() {
        let train_x = array![[8.],
                             [3.],
                             [1.],
                             [4.],
                             [5.],
                             [7.],
                             [2.],
                             [6.]];
        let train_y = array![0, 0, 0, 1, 0, 1, 1, 2];
        let x = array![0.];
        let mut max_k = 8;
        
        let knn = NearestNeighbors::from_data(&x.view(), &train_x.view(),
                                        &train_y.view(), max_k,
                                        euclidean_distance);
        
        let distances_from_x: Vec<_> = knn.neighbors.iter()
                                            .map(|neigh| neigh.distance)
                                            .collect();
        assert_eq!(distances_from_x, vec![1., 2., 3., 4., 5., 6., 7., 8.]);

        // Reduce max_k.
        max_k = 5;
        let knn = NearestNeighbors::from_data(&x.view(), &train_x.view(),
                                        &train_y.view(), max_k,
                                        euclidean_distance);
        
        let distances_from_x: Vec<_> = knn.neighbors.iter()
                                            .map(|neigh| neigh.distance)
                                            .collect();
        assert_eq!(distances_from_x, vec![1., 2., 3., 4., 5.]);
    }

    #[test]
    fn knn_predictions_ties() {
        let train_x = array![[0.], [1.], [1.], [1.], [1.], [1.], [1.], [2.], [2.]];
        let train_y = array![0, 1, 1, 1, 0, 1, 0, 0, 0];

        let x = array![0.];
        let max_k = 10;

        let knn = NearestNeighbors::from_data(&x.view(), &train_x.view(),
                                        &train_y.view(), max_k,
                                        euclidean_distance);

        assert_eq!(knn.predict(1), Ok(0));
        assert_eq!(knn.predict(3), Ok(1));
        assert_eq!(knn.predict(5), Ok(1));

        // The same should happen if examples appear in a different
        // order.
        let train_x = array![[1.], [1.], [1.], [1.], [1.], [1.], [2.], [2.], [0.]];
        let train_y = array![0, 1, 1, 1, 0, 1, 0, 0, 0];

        let knn = NearestNeighbors::from_data(&x.view(), &train_x.view(),
                                        &train_y.view(), max_k,
                                        euclidean_distance);

        assert_eq!(knn.predict(1), Ok(0));
        assert_eq!(knn.predict(3), Ok(1));
        assert_eq!(knn.predict(5), Ok(1));

    }

    #[test]
    fn knn_predictions_multivariate() {

        let train_x = array![[1., 3.],
                             [1., 2.],
                             [2., 3.],
                             [2., 2.],
                             [3., 2.],
                             [2., 2.],
                             [2., 2.]];
        let train_y = array![0, 0, 0, 1, 1, 2, 2];

        let x1 = array![2., 2.];
        let x2 = array![2., 2.];

        let max_k = 10;
        let distance = euclidean_distance;

        let mut knn1 = NearestNeighbors::from_data(&x1.view(), &train_x.view(),
                                                   &train_y.view(), max_k,
                                                   distance);
        let knn2 = NearestNeighbors::from_data(&x2.view(), &train_x.view(),
                                              &train_y.view(), max_k,
                                              distance);

        assert_eq!(knn1.predict(1), Ok(2));

        knn1.add_example(&array![2., 1.].view(), 2, distance);

        assert_eq!(knn1.predict(1), Ok(2));
        assert_eq!(knn1.predict(3), Ok(2));
        assert!(knn1.predict(5) == Ok(2) || knn1.predict(5) == Ok(0));

        assert_eq!(knn2.predict(1), Ok(2));
        assert_eq!(knn2.predict(3), Ok(2));
        assert!(knn2.predict(5) == Ok(2) || knn2.predict(5) == Ok(1));
    }

    #[test]
    fn knn_forward_predictions() {
        let train_x = array![[8.],
                             [7.],
                             [6.],
                             [5.],
                             [4.],
                             [3.],
                             [2.],
                             [1.]];
        let train_y = array![0, 0, 0, 1, 0, 1, 1, 2];
        let x = array![0.];
        let max_k = 8;
        let distance = euclidean_distance;

        // NNs of x.
        let mut knn = NearestNeighbors::new(&x.view(), max_k, distance);


        let expected_preds_1 = vec![Ok(0), Ok(0), Ok(0), Ok(1), Ok(0), Ok(1),
                                    Ok(1), Ok(2)];
        let expected_preds_3 = vec![Err(()), Err(()), Ok(0), Ok(0), Ok(0),
                                    Ok(1), Ok(1), Ok(1)];
        let expected_preds_5 = vec![Err(()), Err(()), Err(()), Err(()), Ok(0),
                                    Ok(0), Ok(1), Ok(1)];
        let expected_preds_7 = vec![Err(()), Err(()), Err(()), Err(()),
                                    Err(()), Err(()), Ok(0), Ok(1)];

        for (i, (x, y)) in train_x.outer_iter().zip(train_y.iter()).enumerate() {
            knn.add_example(&x, *y, distance);
            assert_eq!(knn.predict(1), expected_preds_1[i]);
            assert_eq!(knn.predict(3), expected_preds_3[i]);
            assert_eq!(knn.predict(5), expected_preds_5[i]);
            assert_eq!(knn.predict(7), expected_preds_7[i]);
        }
    }

    #[test]
    fn knn_forward_errors() {
        let train_x = array![[8.],
                             [7.],
                             [6.],
                             [5.],
                             [4.],
                             [3.],
                             [2.],
                             [1.]];
        let train_y = array![0, 0, 0, 1, 0, 1, 1, 2];
        let test_x = array![[3.],
                            [0.],
                            [6.],
                            [1.],
                            [6.],
                            [4.],
                            [5.]];
        let test_y = array![0, 0, 2, 1, 0, 1, 0];
        let max_k = 8;
        let distance = euclidean_distance;

        // Test for k = 1.
        let k = 1;
        let mut knn = KNNEstimator::new(&test_x.view(), &test_y.view(), k,
                                        max_k, distance);

        // FIXME: I'm not sure why in this case, differently from the
        // backward test, I need to include one more error and prediction.
        // The rest is exactly identical.
        let expected_preds = vec![[1, 2, 0, 2, 0, 0, 1],
                                  [1, 1, 0, 1, 0, 0, 1],
                                  [1, 1, 0, 1, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 0, 1, 0, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0]];
        let expected_error = vec![0.8571428571428571, 0.7142857142857143,
                                  0.7142857142857143, 0.5714285714285714,
                                  0.5714285714285714, 0.42857142857142855,
                                  0.42857142857142855, 0.42857142857142855];
        // NOTE: expected_error correspond to the following error counts:
        // [6, 5, 5, 4, 4, 3, 3]

        for (i, (x, y)) in train_x.outer_iter().zip(train_y.iter()).enumerate() {
            knn.add_example(&x, *y, distance).unwrap();
            assert_eq!(knn.predictions, expected_preds[expected_preds.len()-1-i]);
            assert_eq!(knn.get_error(), expected_error[expected_error.len()-1-i]);
        }

        // Test when changing k.
        let k = 1;
        let max_k = train_y.len();
        let mut knn = KNNEstimator::new(&test_x.view(), &test_y.view(), k,
                                        max_k, distance);
        let expected_error = vec![0.42857142857142855, 0.42857142857142855,
                                  0.42857142857142855, 0.5714285714285714,
                                  0.42857142857142855, 0.42857142857142855,
                                  0.42857142857142855, 0.42857142857142855];
        let expected_preds = vec![vec![0; 7], vec![0; 7], vec![0; 7],
                                  //NOTE: the prediction vector right
                                  //below this comment may also be
                                  //vec![1, 1, 1, 1, 0, 1, 0].
                                  vec![1, 1, 0, 1, 0, 1, 1],
                                  vec![0; 7],
                                  vec![1, 1, 0, 1, 0, 1, 0],
                                  vec![1, 1, 0, 1, 0, 1, 0],
                                  vec![1, 1, 0, 1, 0, 1, 0]];
        let ks = vec![1, 1, 1, 1, 3, 3, 5, 5];

        for (i, (x, y)) in train_x.outer_iter().zip(train_y.iter()).enumerate() {
            knn.set_k(ks[i]).unwrap();
            knn.add_example(&x, *y, distance).unwrap();
            assert_eq!(knn.get_error(), expected_error[i]);
            assert_eq!(knn.predictions, expected_preds[i]);
        }
    }

    #[test]
    fn ties_after_max_k() {
        let max_k = 5;
        let distance = euclidean_distance;

        let train_x = array![[0.], [0.], [0.], [1.], [1.], [1.], [1.], [1.], [0.]];
        let train_y = array![0, 1, 1, 1, 0, 0, 0, 0, 1];

        let x = array![0.];
        let mut nn = NearestNeighbors::from_data(&x.view(), &train_x.view(),
                                                 &train_y.view(), max_k,
                                                 distance);
        let distances_from_x: Vec<_> = nn.neighbors.iter()
                                            .map(|neigh| neigh.distance)
                                            .collect();
        assert_eq!(distances_from_x, vec![0., 0., 0., 0., 1.]);
        assert_eq!(nn.extra_ties_dist, Some(1.));
        println!("{:?}", nn.extra_ties);
        // NOTE: equally valid would be to have extra_ties[0] == Some(4)
        // and extra_ties[1] == None.
        assert_eq!(nn.extra_ties.get(&0), Some(&3));
        assert_eq!(nn.extra_ties.get(&1), Some(&1));

        assert_eq!(nn.predict(3).unwrap(), 1);
        assert_eq!(nn.predict(5).unwrap(), 1);
        assert!(nn.predict(6).is_err());

        // Test ties count again.
        nn.add_example(&array![1.].view(), 0, distance);
        nn.add_example(&array![1.].view(), 1, distance);
        nn.add_example(&array![1.].view(), 2, distance);
        nn.add_example(&array![1.].view(), 2, distance);
        nn.add_example(&array![1.].view(), 3, distance);

        assert_eq!(nn.extra_ties.get(&0), Some(&4));
        assert_eq!(nn.extra_ties.get(&1), Some(&2));
        assert_eq!(nn.extra_ties.get(&2), Some(&2));
        assert_eq!(nn.extra_ties.get(&3), Some(&1));

        // Different labels.
        let train_y = array![0, 0, 1, 1, 0, 0, 0, 0, 1];
        let nn = NearestNeighbors::from_data(&x.view(), &train_x.view(),
                                             &train_y.view(), max_k,
                                             euclidean_distance);
        assert_eq!(nn.predict(5).unwrap(), 0);
    }

    #[test]
    /// KNNEstimator's parameter updated_k should take ties into account.
    fn test_updated_k() {
        let test_x = array![[0.]];
        let test_y = array![1];

        let k = 1;
        let max_k = 5;  // Essential that max_k > k for this test, otherwise
                        // it's a different check.
        let distance = euclidean_distance;

        let mut knn = KNNEstimator::new(&test_x.view(), &test_y.view(), k,
                                        max_k, euclidean_distance);

        // We'll only observe examples with distance 2 from x.
        println!("asdf");

        // First all with label 0.
        for _ in 0..5 {
            knn.add_example(&array![2.].view(), 0, distance).unwrap();
            println!("asdf");
        }

        assert_eq!(knn.predictions, vec![0]);

        // Now we change the ties' label distribution to 1.
        for _ in 0..6 {
            knn.add_example(&array![2.].view(), 1, distance).unwrap();
            println!("asdf");
        }

        assert_eq!(knn.predictions, vec![1]);
    }
}
