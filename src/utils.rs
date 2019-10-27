//! Utility routines for loading and storing data into files.
extern crate csv;

use ndarray::prelude::*;
use std::error::Error;
use std::collections::HashMap;
// FIXME: not sure how to remove "self" from the line below.
use self::csv::ReaderBuilder;
use std::str::FromStr;
use std::f64;

use fbleau::Label;

/// Loads a CSV data file.
///
/// The file format should be, for each row:
///     label, x1, x2, ...
/// where x1, x2, ... are features forming a feature vector.
pub fn load_data<T>(fname: &str)
        -> Result<(Array2<T>, Array1<Label>), Box<dyn Error>>
        where T: FromStr {
    let mut reader = ReaderBuilder::new()
                                   .has_headers(false)
                                   .from_path(fname)?;

    let mut inputs: Vec<T> = Vec::new();
    let mut targets: Vec<Label> = Vec::new();
    let mut ncols: Option<usize> = None;

    for result in reader.records() {
        let record = result?;

        inputs.extend(record.iter()
                            .skip(1)  // First one is the label.
                            .map(|x| x.trim()
                                      .parse::<T>().ok()
                                                     .expect("Failed to parse")));
        targets.push(record[0].parse::<usize>()
                .unwrap_or_else(|_|
                    panic!(format!("Could not parse file {}. Error at line: {:?}",
                                   fname, record))));

        if let Some(x) = ncols {
            if x != record.len() - 1 {
                panic!("File has wrong format");
            }
        } else {
            ncols = Some(record.len() - 1);
        }
    }

    let inputs_a = if let Some(d) = ncols {
        let n = inputs.len() / d;
        Array::from_vec(inputs)
              .into_shape((n, d))?
    } else {
        panic!("File has wrong format");
    };

    Ok((inputs_a, Array::from_vec(targets)))
}

/// Prepare training and evaluation data.
///
/// It makes sure the labels are indexed starting from 0,
/// and, if required, it scales the features.
pub fn prepare_data(mut train_x: Array2<f64>, train_y: Array1<Label>,
                    mut test_x: Array2<f64>, test_y: Array1<Label>,
                    scale: bool) -> (Array2<f64>, Array1<Label>,
                                     Array2<f64>, Array1<Label>, usize) {
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
                Each test label should appear in the training data;
                the converse is not necessary");

    // Scale features.
    if train_x.cols() > 1 && scale {
        println!("scaling features");
        scale01(&mut train_x);
        scale01(&mut test_x);
    }

    (train_x, train_y, test_x, test_y, nlabels)
}

/// Returns true if all the elements of the vector
/// can be converted into integers without loss.
///
/// # Examples
///
/// ```
/// assert!(has_integer_support(vec![3, 6, 0, -4]));
/// assert_ne!(has_integer_support(vec![2, 5.5, 3]);
/// ```
pub fn has_integer_support(v: &Array2<f64>) -> bool {
    for x in v.iter() {
        if x.fract() != 0. {
            return false;
        }
    }
    true
}

/// Represents d-dimensional vector objects into 1-dimentional
/// unique ids.
pub fn vectors_to_ids(objects: ArrayView2<usize>,
        mapping: Option<HashMap<Array1<usize>, usize>>)
        -> (Array1<usize>, HashMap<Array1<usize>, usize>) {
    let mut out = Vec::with_capacity(objects.rows());

    let mut next_id = 0;
    let mut mapping = if let Some(mapping) = mapping {
        if let Some(id) = mapping.values().max() {
            next_id = id + 1;
        }
        mapping
    } else {
        HashMap::new()
    };

    for x in objects.outer_iter() {
        let id = mapping.entry(x.to_owned())
                        .or_insert_with(|| { next_id += 1; next_id-1 });
        out.push(*id);
    }

    (Array::from_vec(out), mapping)
}

/// Scales columns' values in [0,1] with min-max scaling.
pub fn scale01(matrix: &mut Array2<f64>) {
    let mut max = Array::ones(matrix.cols()) * -f64::INFINITY;
    let mut min = Array::ones(matrix.cols()) * f64::INFINITY;

    for row in matrix.outer_iter() {
        for i in 0..row.len() {
            if min[i] > row[i] {
               min[i] = row[i];
            }
            if max[i] < row[i] {
                max[i] = row[i];
            }
        }
    }

    for mut row in matrix.outer_iter_mut() {
        for i in 0..row.len() {
            row[i] = (row[i] - min[i]) / (max[i] - min[i]);
        }
    }
}

/// Estimates the priors on a vector of labels, and computes the
/// random guessing error as 1 - max priors.
pub fn estimate_random_guessing(labels: &ArrayView1<usize>) -> f64 {
    let mut counts = HashMap::new();
    let mut max_count = 0;

    for y in labels {
        let count = counts.entry(y).or_insert(0);
        *count += 1;
        if *count > max_count {
            max_count = *count;
        }
    }

    1. - f64::from(max_count) / (labels.len() as f64)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::*;

    #[test]
    fn test_vectors_to_ids() {
        let a = array![[1, 2, 3],
                       [3, 4, 5],
                       [6, 4, 3],
                       [1, 2, 3],
                       [6, 4, 3],
                       [6, 4, 3],
                       [9, 0, 2],
                       [9, 0, 2],
                       [6, 4, 3]];
        let (ids_a, mapping) = vectors_to_ids(a.view(), None);
        assert_eq!(ids_a, array![0, 1, 2, 0, 2, 2, 3, 3, 2]);

        // Call for a second array.
        let b = array![[9, 0, 2],
                       [1, 2, 3],
                       [0, 0, 0],
                       [1, 2, 5],
                       [6, 4, 3]];
        let (ids_b, _) = vectors_to_ids(b.view(), Some(mapping));
        assert_eq!(ids_b, array![3, 0, 4, 5, 2]);
    }

    #[test]
    fn test_scale() {
        let mut a = array![[2., 3., 5.],
                           [1., 2., 10.],
                           [0., 1., 2.]];

        scale01(&mut a);

        assert_eq!(a, array![[1. ,1. ,0.375],
                             [0.5, 0.5, 1.],
                             [0., 0., 0.]]);
    }
}
