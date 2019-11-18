#[macro_use]
extern crate bencher;
#[macro_use]
extern crate ndarray;
extern crate rustlearn;

extern crate fbleau;

use fbleau::*;
use fbleau::estimates::*;
use fbleau::estimates::knn::KNNEstimator;
use bencher::Bencher;
use ndarray::prelude::*;
use rustlearn::datasets::*;



/// Load the Boston dataset.
///
/// NOTE: we don't really care about preserving the original data;
/// in fact, the labels in the original data have floating point
/// values, but we convert them into `Label`.
/// This dataset only serves for benchmark purposes.
fn load_boston() -> (Array2<f64>, Array1<Label>, Array2<f64>, Array1<Label>) {
    let n = 506;
    let n_train = 406;
    let d = 13;

    let (data, target) = boston::load_data();
    let train_y = Array::from(target.data().iter()
                                    .take(n_train)
                                    .map(|y| *y as Label)
                                    .collect::<Vec<_>>());
    let test_y = Array::from(target.data().iter()
                                   .skip(n_train)
                                   .map(|y| *y as Label)
                                    .collect::<Vec<_>>());
    let train_x = Array::from_shape_vec((n_train, d), data.data().iter()
                                                           .take(n_train*d)
                                                           .map(|x| *x as f64)
                                                           .collect()).unwrap();
    let test_x = Array::from_shape_vec((n-n_train, d), data.data().iter()
                                                           .skip(n_train*d)
                                                           .map(|x| *x as f64)
                                                           .collect()).unwrap();

    (train_x, train_y, test_x, test_y)
}

/// Load the Iris dataset.
#[allow(dead_code)]
fn load_iris() -> (Array2<f64>, Array1<Label>, Array2<f64>, Array1<Label>) {
    let n = 150;
    let n_train = 100;
    let d = 4;

    let (data, target) = iris::load_data();
    let train_y = Array::from(target.data().iter()
                                    .take(n_train)
                                    .map(|y| *y as Label)
                                    .collect::<Vec<_>>());
    let test_y = Array::from(target.data().iter()
                                   .skip(n_train)
                                   .map(|y| *y as Label)
                                    .collect::<Vec<_>>());
    let train_x = Array::from_shape_vec((n_train, d), data.data().iter()
                                                           .take(n_train*d)
                                                           .map(|x| *x as f64)
                                                           .collect()).unwrap();
    let test_x = Array::from_shape_vec((n-n_train, d), data.data().iter()
                                                           .skip(n_train*d)
                                                           .map(|x| *x as f64)
                                                           .collect()).unwrap();

    (train_x, train_y, test_x, test_y)
}

fn bench_knn_forward(b: &mut Bencher) {
    let (train_x, train_y, test_x, test_y) = load_boston();
    let n_train = train_x.nrows();

    b.iter(|| {
        let mut knn = KNNEstimator::new(&test_x.view(), &test_y.view(),
                                        n_train, euclidean_distance,
                                        KNNStrategy::Ln);
        for (n, (x, y)) in train_x.outer_iter().zip(train_y.iter()).enumerate() {
            let k = if n != 0 {
                let k = (n as f64).ln().ceil() as usize;
                if k % 2 == 0 { k + 1 } else { k }
            } else {
                1
            };
            knn.set_k(k).expect("Failed when changing k");
            knn.add_example(&x, *y).expect("Failed to add new example");
        }
    });
}

benchmark_group!(benches, bench_knn_forward);
benchmark_main!(benches);
