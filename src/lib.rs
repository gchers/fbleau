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
//! F-BLEAU is thought to be mainly used via the binary it provides, `fbleau`.
//! For usage instructions, please refer to
//! [fbleau's home page](https://github.com/gchers/fbleau)
//! or to the help screen: `fbleau -h`.
//!
//! For the library documentation, please refer to the appropriate links
//! within this page.
//!
//! # References
//!
//! [1] 2017, "Bayes, not Naïve: Security Bounds on Website Fingerprinting Defenses". _Giovanni Cherubin_
//!
//! [2] 2018, "F-BLEAU: Practical Channel Leakage Estimation". _Giovanni Cherubin, Konstantinos Chatzikokolakis, Catuscia Palamidessi_.
//!
//! [3] (Blog) "Machine Learning methods for Quantifying the Security of Black-boxes". https://giocher.com/pages/bayes.html
extern crate csv;
extern crate ndarray;
#[macro_use]
extern crate itertools;
extern crate ndarray_parallel;
extern crate ordered_float;
extern crate float_cmp;
extern crate strsim;

pub mod estimates;

pub type Label = usize;
