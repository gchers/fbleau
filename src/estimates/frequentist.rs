//! Frequentist Bayes risks estimate for discrete secret and output space.
use ndarray::*;
use itertools::Itertools;
use std::collections::HashMap;

use Label;
use estimates::{some_or_error};

// Type of the elements of a feature vector.
type ObjectValue = usize;

/// Represents the frequencies of some observation.
///
/// It will be used both for priors P(y) and for the joint distribution
/// P(x, y) for each x.
#[derive(Debug)]
struct FrequencyCount {
    count: Vec<usize>,
    prediction: Option<Label>,
}

impl FrequencyCount {
    fn new(n_labels: usize) -> FrequencyCount {
        FrequencyCount {
            count: vec![0; n_labels],
            prediction: None,
        }
    }

    /// Returns either `Some` prediction, or `None` (in case there's
    /// no count information).
    fn predict(&self) -> Option<Label> {
        self.prediction
    }

    /// Increases the count for an observed label `y`, and changes
    /// the prediction if needed.
    fn add_example(&mut self, y: Label) -> bool {
        self.count[y] += 1;
        let mut updated = false;

        if let Some(pred) = self.prediction {
            if y != pred {
                // Did the maximum prior change?
                if self.count[y] > self.count[pred] {
                    self.prediction =  Some(y);
                    updated = true;
                }
            }
        }
        else {
            self.prediction = Some(y);
            updated = true;
        }
        updated
    }

    /// Removes one observed label `y`, and changes the prediction
    /// accordingly.
    ///
    /// `true` is returned if the prediction changed, `false`, otherwise.
    fn remove_example(&mut self, y: Label) -> bool {
        self.count[y] -= 1;

        // Don't need to change prediction if y wasn't the predicted
        // label before.
        if Some(y) == self.prediction {
            // Check if there's a more likely prediction.
            let mut new_pred = y;
            for (yi, &c) in self.count.iter().enumerate() {
                if c > self.count[y] {
                    new_pred = yi;
                    // If this count is larger than the count for the
                    // original prediction, then necessarily it is also
                    // the largest count right now, so we can stop here.
                    //break;
                }
            }

            if self.count[new_pred] == 0 {
                // We have no more information.
                self.prediction = None;
                return true;
            }

            if self.prediction != Some(new_pred) {
                self.prediction = Some(new_pred);
                return true;
            }
        }
        false
    }
}

/// Keeps track of the frequentist estimate, allowing to reduce the
/// size of training data.
pub struct FrequentistEstimator {
    // Keeps the count of each label y associated with each object x.
    joint_count: HashMap<ObjectValue, FrequencyCount>,
    // Keeps the count of each label.
    priors_count: FrequencyCount,
    // Bayes risk estimate.
    error_count: usize,
    // NOTE: at the moment I'm not sure how we could remove training and
    // test data from here.
    // Of course, it'd be possible to do so with an increasing training set
    // strategy, but here we're removing from the training set.
    train_x: Vec<ObjectValue>,
    train_y: Vec<Label>,
    test_x: Vec<ObjectValue>,
    test_y: Vec<Label>,
}

impl FrequentistEstimator {
    pub fn new(n_labels: usize, test_x: &ArrayView1<ObjectValue>,
               test_y: &ArrayView1<Label>)
            -> FrequentistEstimator {

        // Init counts.
        let priors_count = FrequencyCount::new(n_labels);
        let mut joint_count: HashMap<ObjectValue, FrequencyCount> = HashMap::new();

        // Instantiate points for which we need a prediction.
        // We'll only have information for the intersection of
        // train_x and test_x; for the others we'll have to guess
        // according to priors.
        for &x in test_x.iter().unique() {
            joint_count.entry(x)
                       .or_insert(FrequencyCount::new(n_labels));
        }

        FrequentistEstimator {
            joint_count: joint_count,
            priors_count: priors_count,
            error_count: 0,
            train_x: vec![],
            train_y: vec![],
            test_x: test_x.to_vec(),
            test_y: test_y.to_vec(),
        }
    }



    pub fn from_data(n_labels: usize, train_x: &ArrayView1<ObjectValue>,
                     train_y: &ArrayView1<Label>, test_x: &ArrayView1<ObjectValue>,
                     test_y: &ArrayView1<Label>)
            -> FrequentistEstimator {

        // FIXME: instantiate from new().
        // Init counts.
        let mut joint_count: HashMap<ObjectValue, FrequencyCount> = HashMap::new();
        let mut priors_count = FrequencyCount::new(n_labels);

        // Instantiate points for which we need a prediction.
        // We'll only have information for the intersection of
        // train_x and test_x; for the others we'll have to guess
        // according to priors.
        for &x in test_x.iter().unique() {
            joint_count.entry(x)
                       .or_insert(FrequencyCount::new(n_labels));
        }


        // Count frequencies in training data.
        for (x, &y) in train_x.iter().zip(train_y) {
            assert!(y < n_labels,
                "labels' values must be < number of labels");
            priors_count.add_example(y);
            if let Some(jx) = joint_count.get_mut(x) {
                jx.add_example(y);
            }
        }

        // Compute Bayes risk.
        let mut error_count = 0;

        for (x, &y) in test_x.iter().zip(test_y) {
            let jx = joint_count.get(x)
                          .expect("shouldn't happen");

            let pred = match jx.predict() {
                Some(pred) => pred,
                None => priors_count.predict().expect("not enough info for priors"),
            };

            if y != pred {
                error_count += 1;
            }
        }

        FrequentistEstimator {
            joint_count: joint_count,
            priors_count: priors_count,
            error_count: error_count,
            train_x: train_x.to_vec(),
            train_y: train_y.to_vec(),
            test_x: test_x.to_vec(),
            test_y: test_y.to_vec(),
        }
    }

    /// Updates the predictions when the very first example is added,
    /// and therefore we don't even have any information on priors.
    fn add_first_example(&mut self, x: ObjectValue, y: Label) {
        self.error_count = 0;

        self.priors_count.add_example(y);
        let pred = y;

        if let Some(jx) = self.joint_count.get_mut(&x) {
            jx.add_example(y);
        }

        for yi in &self.test_y {
            let error = if *yi != pred { 1 } else { 0 };
            self.error_count += error;
        }
    }


    pub fn add_example(&mut self, x: ObjectValue, y: Label) {
        self.train_x.push(x);
        self.train_y.push(y);

        let mut old_priors_pred = match self.priors_count.predict() {
            Some(pred) => pred,
            None => { return self.add_first_example(x, y) },
        };

        // If max prior changed, update predictions for those that were
        // predicted with priors.
        let priors_changed = self.priors_count.add_example(y);
        if priors_changed {
            let new_pred = self.priors_count.predict().unwrap();

            for (xi, &yi) in self.test_x.iter().zip(&self.test_y) {
                // Match points for which we random guess.
                let joint = self.joint_count.get(xi).expect("shouldn't happen");
                if joint.predict().is_none() {
                    let old_error = if yi != old_priors_pred { 1 } else { 0 };
                    let new_error = if yi != new_pred { 1 } else { 0 };

                    self.error_count = self.error_count + new_error - old_error;
                }
            }
            // NOTE: we also need to update the value of old_priors_pred,
            // because otherwise we'll have issues when updating w.r.t.
            // the joint distribution later in this function.
            old_priors_pred = new_pred;
        }

        // Update joint counts (and error), but only if `x` appears
        // in the test set.
        if let Some(joint) = self.joint_count.get_mut(&x) {
            let old_pred = match joint.predict() {
                Some(pred) => pred,
                None => old_priors_pred,
            };
            // Only update prediction if max P(o,s) changed.
            let joint_changed = joint.add_example(y);
            if joint_changed {
                // Predict again.
                let new_pred = joint.predict().unwrap();

                for (&xi, &yi) in self.test_x.iter().zip(&self.test_y) {
                    // Only update predictions for observations with value `x`.
                    if xi == x {
                        let old_error = if yi != old_pred { 1 } else { 0 };
                        let new_error = if yi != new_pred { 1 } else { 0 };

                        self.error_count = self.error_count + new_error - old_error;
                    }
                }
            }
        }
    }

    pub fn remove_one(&mut self) -> Result<(), ()> {
        // TODO: better error handling.
        let x = some_or_error(self.train_x.pop())?;
        let y = some_or_error(self.train_y.pop())?;


        // Update priors and if they changed update the error count.
        let old_priors_pred = some_or_error(self.priors_count.predict())?;
        let priors_changed = self.priors_count.remove_example(y);

        if priors_changed {
            // Predict again those that were predicted with priors.
            let new_pred = some_or_error(self.priors_count.predict())?;

            for (xi, &yi) in self.test_x.iter().zip(&self.test_y) {
                // Match points for which we random guess.
                let joint = self.joint_count.get(xi).expect("shouldn't happen");
                if joint.predict().is_none() {
                    let old_error = if yi != old_priors_pred { 1 } else { 0 };
                    let new_error = if yi != new_pred { 1 } else { 0 };

                    self.error_count = self.error_count + new_error - old_error;
                }
            }
        }

        // Update joint counts (and error), but only if `x` appears
        // in the test set.
        if let Some(joint) = self.joint_count.get_mut(&x) {
            let old_joint_pred = joint.predict()
                                      .expect("shouldn't fail here");
            let joint_changed = joint.remove_example(y);

            if joint_changed {
                // Predict again.
                let new_pred = match self.priors_count.predict() {
                    Some(pred) => pred,
                    // This means we don't have any more information on
                    // P(x, y), and we'll need to predict via priors.
                    None => some_or_error(self.priors_count.predict())?,
                };

                for (&xi, &yi) in self.test_x.iter().zip(&self.test_y) {
                    if xi == x {
                        let old_error = if yi != old_joint_pred { 1 } else { 0 };
                        let new_error = if yi != new_pred { 1 } else { 0 };

                        self.error_count = self.error_count + new_error - old_error;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get_error(&self) -> f64 {
        (self.error_count as f64) / (self.test_y.len() as f64)
    }

    pub fn get_error_count(&self) -> usize {
        self.error_count
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequentist_init() {
        let n_labels = 3;
        let train_x = array![0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 6];
        let train_y = array![0, 1, 1, 2, 2, 1, 2, 2, 0, 1, 1, 1];

        let test_x = array![0, 0, 1, 2, 2, 8, 8];
        let test_y = array![1, 1, 1, 1, 2, 1, 0];

        let freq = FrequentistEstimator::from_data(n_labels,
                                                   &train_x.view(),
                                                   &train_y.view(),
                                                   &test_x.view(),
                                                   &test_y.view());

        // Only keeps track of (unique) points that are in train_x.
        assert_eq!(freq.joint_count.len(), 4);

        // Priors counts.
        assert_eq!(freq.priors_count.count, vec![2, 6, 4]);
        // Joint probability counts for objects 0, 1, 2, 8.
        assert_eq!(freq.joint_count.get(&0).unwrap().count, vec![1, 2, 0]);
        assert_eq!(freq.joint_count.get(&1).unwrap().count, vec![0, 0, 2]);
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![1, 3, 2]);
        assert_eq!(freq.joint_count.get(&8).unwrap().count, vec![0; 3]);

        // Individual predictions.
        assert_eq!(freq.joint_count.get(&0).unwrap().predict().unwrap(), 1);
        assert_eq!(freq.joint_count.get(&1).unwrap().predict().unwrap(), 2);
        assert_eq!(freq.joint_count.get(&2).unwrap().predict().unwrap(), 1);
        assert!(freq.joint_count.get(&8).unwrap().predict().is_none());

        //// Estimate.
        assert_eq!(freq.error_count, 3);
        assert_eq!(freq.get_error(), 3./7.);
    }

    #[test]
    fn frequentist_estimate_backward() {
        let n_labels = 3;
        let train_x = array![0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 6];
        let train_y = array![0, 1, 1, 2, 2, 1, 2, 2, 0, 1, 1, 1];

        let test_x = array![0, 0, 1, 2, 2, 8, 8];
        let test_y = array![1, 1, 1, 1, 2, 1, 0];

        let mut freq = FrequentistEstimator::from_data(n_labels,
                                                       &train_x.view(),
                                                       &train_y.view(),
                                                       &test_x.view(),
                                                       &test_y.view());

        // Estimate.
        // 0)
        assert_eq!(freq.error_count, 3);
        assert_eq!(freq.priors_count.count, vec![2, 6, 4]);

        // 1)
        freq.remove_one().unwrap();
        assert_eq!(freq.priors_count.count, vec![2, 5, 4]);
        assert_eq!(freq.error_count, 3);

        // 2)
        freq.remove_one().unwrap();
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![1, 2, 2]);
        let pred = freq.joint_count.get(&2).unwrap().predict().unwrap();
        // More properly it should be: assert!(pred == 1 || pred == 2);
        assert_eq!(pred, 1);
        assert_eq!(freq.priors_count.count, vec![2, 4, 4]);
        assert_eq!(freq.error_count, 3);

        // 3)
        freq.remove_one().unwrap();
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![1, 1, 2]);
        assert_eq!(freq.joint_count.get(&2).unwrap().predict().unwrap(), 2);
        assert_eq!(freq.priors_count.count, vec![2, 3, 4]);
        assert_eq!(freq.priors_count.predict().unwrap(), 2);
        assert_eq!(freq.error_count, 4);    // Increases because of priors.

        // 4)
        freq.remove_one().unwrap();
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![0, 1, 2]);
        assert_eq!(freq.joint_count.get(&2).unwrap().predict().unwrap(), 2);
        assert_eq!(freq.priors_count.count, vec![1, 3, 4]);
        assert_eq!(freq.error_count, 4);

        // 5)
        freq.remove_one().unwrap();
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![0, 1, 1]);
        assert_eq!(freq.joint_count.get(&2).unwrap().predict().unwrap(), 2);
        assert_eq!(freq.priors_count.count, vec![1, 3, 3]);
        assert_eq!(freq.error_count, 4);

        // 6)
        freq.remove_one().unwrap();
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![0, 1, 0]);
        assert_eq!(freq.joint_count.get(&2).unwrap().predict().unwrap(), 1);
        assert_eq!(freq.priors_count.count, vec![1, 3, 2]);
        assert_eq!(freq.error_count, 3);

        // 7)
        freq.remove_one().unwrap();
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![0, 0, 0]);
        // Starts predicting with priors also for 2.
        assert!(freq.joint_count.get(&2).unwrap().predict().is_none());
        assert_eq!(freq.priors_count.count, vec![1, 2, 2]);
        assert_eq!(freq.error_count, 3);

        // 8)
        freq.remove_one().unwrap();
        assert_eq!(freq.joint_count.get(&1).unwrap().count, vec![0, 0, 1]);
        assert_eq!(freq.joint_count.get(&1).unwrap().predict().unwrap(), 2);
        assert_eq!(freq.priors_count.count, vec![1, 2, 1]);
        assert_eq!(freq.error_count, 3);

        // 9)
        freq.remove_one().unwrap();
        assert_eq!(freq.joint_count.get(&1).unwrap().count, vec![0, 0, 0]);
        assert!(freq.joint_count.get(&1).unwrap().predict().is_none());
        assert_eq!(freq.priors_count.count, vec![1, 2, 0]);
        assert_eq!(freq.error_count, 2);

        // 10)
        freq.remove_one().unwrap();
        assert_eq!(freq.joint_count.get(&0).unwrap().count, vec![1, 1, 0]);
        //assert_eq!(freq.joint_count.get(&0).unwrap().predict().unwrap(), 0);
        assert_eq!(freq.priors_count.count, vec![1, 1, 0]);
        assert_eq!(freq.error_count, 2);

        // 11)
        freq.remove_one().unwrap();
        assert_eq!(freq.joint_count.get(&0).unwrap().count, vec![1, 0, 0]);
        assert_eq!(freq.joint_count.get(&0).unwrap().predict().unwrap(), 0);
        assert_eq!(freq.priors_count.count, vec![1, 0, 0]);
        assert_eq!(freq.error_count, 6);

        assert!(freq.remove_one().is_err());
    }

    #[test]
    fn frequentist_estimate_forward() {
        let n_labels = 3;
        let train_x = array![0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 6];
        let train_y = array![0, 1, 1, 2, 2, 1, 2, 2, 0, 1, 1, 1];

        let test_x = array![0, 0, 1, 2, 2, 8, 8];
        let test_y = array![1, 1, 1, 1, 2, 1, 0];

        let mut freq = FrequentistEstimator::new(n_labels,
                                                 &test_x.view(),
                                                 &test_y.view());

        // Estimate.
        // 11)
        freq.add_example(train_x[0], train_y[0]);
        assert_eq!(freq.joint_count.get(&0).unwrap().count, vec![1, 0, 0]);
        assert_eq!(freq.joint_count.get(&0).unwrap().predict().unwrap(), 0);
        assert_eq!(freq.priors_count.count, vec![1, 0, 0]);
        assert_eq!(freq.error_count, 6);

        // 10)
        freq.add_example(train_x[1], train_y[1]);
        assert_eq!(freq.joint_count.get(&0).unwrap().count, vec![1, 1, 0]);
        assert_eq!(freq.priors_count.count, vec![1, 1, 0]);
        //assert_eq!(freq.error_count, 2);
        // The prediction of priors could be either 0 or 1, and so
        // the prediction of joint probability for object 0.
        // However, I'll keep the assertion strict.
        //assert!(freq.error_count == 2 || freq.error_count == 6);
        assert_eq!(freq.error_count, 6);

        // 9)
        freq.add_example(train_x[2], train_y[2]);
        assert_eq!(freq.joint_count.get(&1).unwrap().count, vec![0, 0, 0]);
        assert!(freq.joint_count.get(&1).unwrap().predict().is_none());
        assert_eq!(freq.priors_count.count, vec![1, 2, 0]);
        assert_eq!(freq.error_count, 2);


        // 8)
        freq.add_example(train_x[3], train_y[3]);
        assert_eq!(freq.joint_count.get(&1).unwrap().count, vec![0, 0, 1]);
        assert_eq!(freq.joint_count.get(&1).unwrap().predict().unwrap(), 2);
        assert_eq!(freq.priors_count.count, vec![1, 2, 1]);
        assert_eq!(freq.error_count, 3);

        // 7)
        freq.add_example(train_x[4], train_y[4]);
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![0, 0, 0]);
        // Starts predicting with priors also for 2.
        assert!(freq.joint_count.get(&2).unwrap().predict().is_none());
        assert_eq!(freq.priors_count.count, vec![1, 2, 2]);
        assert_eq!(freq.error_count, 3);

        // 6)
        freq.add_example(train_x[5], train_y[5]);
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![0, 1, 0]);
        assert_eq!(freq.joint_count.get(&2).unwrap().predict().unwrap(), 1);
        assert_eq!(freq.priors_count.count, vec![1, 3, 2]);
        assert_eq!(freq.error_count, 3);

        // 5)
        freq.add_example(train_x[6], train_y[6]);
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![0, 1, 1]);
        //assert_eq!(freq.joint_count.get(&2).unwrap().predict().unwrap(), 2);
        let pred = freq.joint_count.get(&2).unwrap().predict().unwrap();
        // NOTE: pred could either be 2 or 1, but I'll keep the condition
        // strict so that if anything changes we know.
        //assert!(pred == 2 || pred == 1); 
        assert_eq!(pred, 1);
        assert_eq!(freq.priors_count.count, vec![1, 3, 3]);
        // NOTE: the following could be 3 or 4, but I'll set the condition
        // strict.
        //assert!(freq.error_count == 4 | freq.error_count == 3);
        assert!(freq.error_count == 3);

        // 4)
        freq.add_example(train_x[7], train_y[7]);
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![0, 1, 2]);
        assert_eq!(freq.joint_count.get(&2).unwrap().predict().unwrap(), 2);
        assert_eq!(freq.priors_count.count, vec![1, 3, 4]);
        assert_eq!(freq.error_count, 4);

        // 3)
        freq.add_example(train_x[8], train_y[8]);
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![1, 1, 2]);
        assert_eq!(freq.joint_count.get(&2).unwrap().predict().unwrap(), 2);
        assert_eq!(freq.priors_count.count, vec![2, 3, 4]);
        assert_eq!(freq.priors_count.predict().unwrap(), 2);
        assert_eq!(freq.error_count, 4);    // Increases because of priors.

        // 2)
        freq.add_example(train_x[9], train_y[9]);
        assert_eq!(freq.joint_count.get(&2).unwrap().count, vec![1, 2, 2]);
        let pred = freq.joint_count.get(&2).unwrap().predict().unwrap();
        // More properly it should be: assert!(pred == 1 || pred == 2);
        assert_eq!(pred, 2);
        assert_eq!(freq.priors_count.count, vec![2, 4, 4]);
        // Should be assert!(freq.error_count == 3 || freq.error_count == 4);
        assert_eq!(freq.error_count, 4);

        // 1)
        freq.add_example(train_x[10], train_y[10]);
        assert_eq!(freq.priors_count.count, vec![2, 5, 4]);
        assert_eq!(freq.error_count, 3);

        // 0)
        freq.add_example(train_x[11], train_y[11]);
        assert_eq!(freq.error_count, 3);
        assert_eq!(freq.priors_count.count, vec![2, 6, 4]);
    }
}
