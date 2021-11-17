# F-BLEAU <br><img src="https://github.com/gchers/fbleau/raw/gh-pages/img/forest.png" width="150" height="150" />
[![build](https://github.com/gchers/fbleau/actions/workflows/build.yml/badge.svg)](https://github.com/gchers/fbleau/actions/workflows/build.yml) ![Version](https://img.shields.io/crates/v/fbleau.svg)

F-BLEAU is a tool for estimating the information leakage of a (black-box) system,
by only looking at examples of secret inputs and
respective outputs. It considers a generic system as a black-box, taking
secret inputs and returning outputs accordingly, and it measures how much the
outputs "leak" about the inputs.

F-BLEAU is based on the equivalence between estimating the error of a Machine
Learning model of a specific class and the estimation of information leakage.

This code was used for the experiments of [2] on the following
evaluations: Gowalla, e-passport, and side channel attack to finite field
exponentiation.

## Implemented methods

We use the fact that any universally consistent (UC) Machine Learning rule
can be used as an information leakage estimator.
This applies to virtually any system (channel): probabilistic/deterministic, and
with discrete/continuous secret inputs and outputs [2,4].

The channel has to be stateless (i.e., it shouldn't change over time).
For example, if evaluating a privacy mechanism that adds random noise to
the input signal, this means that the distribution of the noise stays the same.

There are *a lot* of UC rules that we can use. These are the main ones (that we're aware of):

| Method | Type | Input domain | Output domain | Proposed in | Implemented in F-BLEAU  |
| ------------- |:----:|:----:|:----:|:----:|:----:|
| Frequentist | Estimate | Discrete | Discrete | [5] | :heavy_check_mark: |
| NN-bound | Lower bound | Discrete | Discrete/Continuous | [1] | :heavy_check_mark: |
| k-NN | Estimate | Discrete | Discrete/Continuous | [1] | :x: |
| NN* | Estimate | Discrete | Discrete | [2] | :heavy_check_mark: | :x: |
| k-NN* | Estimate | Discrete | Discrete/Continuous | [2] | :heavy_check_mark: |
| SVM | Estimate | Discrete | Discrete/Continuous | [4] | :x:|
| Feed-forward Neural Network | Estimate | Discrete | Discrete/Continuous | [4] | :x: |
| k-NN regressor | Estimate | Discrete/Continuous | Discrete/Continuous | [4] | :x: |


*: (k-)NN methods implemented in F-BLEAU are the optimised algorithms proposed in [2]; they
are obtained by extending the Frequentist method in the spirit of the k-NN principle.

Please, do feel free to make a PR with implementations of more methods, or open a ticket
with the methods you'd like to see implemented.


# Getting started

F-BLEAU is provided as a command line tool, `fbleau`.
Python bindings also exist (see below).

`fbleau` takes as input CSV data containing examples of system's inputs
and outputs.
It currently requires two CSV files as input: a _training_ file and a
_validation_ (or _test_) file, such as:

    0, 0.1, 2.43, 1.1
    1, 0.0, 1.22, 1.1
    1, 1.0, 1.02, 0.1
    ...

where the first column specifies the secret, and the remaining ones
indicate the output vector.

It runs a chosen method for estimating the Bayes risk (smallest probability
of error of an adversary at predicting a secret given the respective output),
and relative security measures.

The general syntax is:

```
fbleau <estimate> [--knn-strategy=<strategy>] [options] <train> <eval>

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
    eval                        Validation data (.csv file).
```

## Example

This example considers 100K observations generated according to a
Geometric distribution with privacy level `nu=4` (see [2] for details);
the true value of the Bayes risk is `R*=0.456`, computed analytically.
The observations are split into training (80%) and test sets
(`examples/geometric-4.train.csv` and `examples/geometric-4.test.csv`
respectively).

One can run `fbleau` to compute the `knn` estimate with `ln` strategy
(see below for details about estimation methods) as follows:

```console
$ fbleau knn --knn-strategy ln examples/geometric-4.train.csv examples/geometric-4.test.csv
Random guessing error: 0.913
Estimating leakage measures...

Minimum estimate: 0.473
Multiplicative Leakage: 6.057471264367819
Additive Leakage: 0.44000000000000006
Bayes security measure: 0.5180722891566265
Min-entropy Leakage: 2.5987156557884865
You have new mail in /var/mail/joker

```

NOTE: depending on your machine's specs this may take a while.

By default, F-BLEAU runs the estimator on an increasing number of
training examples, and it computes the estimate at every step.
The returned estimate of R* (here, 0.473) is the smallest one
observed in this process.

To log the estimates at every step, specify a log file with
`--logfile <logfile>`.

## Estimates

In principle, one should try as many estimation methods as possible, and select
the one that produced the smallest estimate [2].
However, some estimators are better indicated for certain cases.
The following table shows: i) when an estimator is guaranteed to converge
to the correct value (provided with enough data), and ii) if they're indicated
for small or large systems.
Indicatively, a small system has up to 1K possible output values; a large system
may have much larger output spaces.

| Estimate | Options | Convergence | Use cases |
| -------- | ---------------- | ----------- | --------- |
| **frequentist** |  | If the output space is finite | Small systems |
| **nn**  |  | If the output space is finite | Small/large systems |
| **knn**  | `--knn-strategy` | Always  | Small/large systems |
| **nn-bound** | | Always (Note, however, that this is a lower bound) | Small/large systems |

For example:
```
fbleau nn <train> <test>
```

Further details are in [2].

### k-NN strategies
k-NN estimators also require defining a "strategy".
Currently implemented strategies are:

**ln** k-NN estimator with `k = ln(n)`, where `n` is the number of training
examples.

**log 10** k-NN estimator with `k = log10(n)`, where `n` is the number of
training examples.

For example, you can run:
```
fbleau knn --knn-strategy log10 <train> <test>
```


## Further options

By default, `fbleau` runs for all training data.
However, one can specify a stopping criterion, in the form of a
(delta, q)-convergence: `fbleau` stops when the estimate's value has
not changed more than delta (`--delta`), either in relative (default) or
absolute (`--absolute`) sense, for at least q steps (`--qstop`).

`fbleau` can scale the individual values of the system's output ("features")
in the `[0,1]` interval by specifying the `--scale` flag.

An option `--distance` is available to select the desired distance metric
for nearest neighbor methods.

Further options are shown in the help page:
```console
fbleau -h
```

# Installation

The code is written in `Rust`, but it is thought to be used as a
standalone command line tool.


Install [rustup](https://rustup.rs), which will make `cargo` available
on your path.
Then run:

```
cargo install fbleau
```

You should now find the binary `fbleau` in your `$PATH` (if not,
check out [rustup](https://rustup.rs) again).

If `rustup` is not available on your system (e.g., some \*BSD systems),
you should still be able to install `cargo` with the system's
package manager, and then install `fbleau` as above.
If this doesn't work, please open a ticket.


# Python bindings

If you prefer using F-BLEAU via Python, we now provide
basic functionalities via a Python module.

To install:
```console
pip install fbleau
```

Usage:
```console
>>> import fbleau
>>> fbleau.run_fbleau(train_x, train_y, test_x, test_y, estimate,
... knn_strategy, distance, logfile, delta, qstop, absolute, scale)
```

Where the parameters follow the above conventions.

```
train_x : training observations (2d numpy array)
train_y : training secrets (1d numpy array)
test_x : test observations (2d numpy array)
test_y : test secrets (1d numpy array)
estimate : estimate, value in ("nn", "knn", "frequentist", "nn-bound")
knn_strategy : if estimate is "knn", specify one in ("ln", "log10")
distance : the distance used for NN or k-NN
log_errors : if `true`, also return the estimate's value (error)
             for each step
log_individual_errors : if `true`, log the individual errors for each
                        test object, for the best estimator
                        (i.e., for the smallest error estimate)
delta : use to stop fbleau when it reaches (delta, qstop)-convergence
qstop : use to stop fbleau when it reaches (delta, qstop)-convergence
absolute : measure absolute instead of relative convergence
scale : scale observations' features in [0,1]
```

The function `run_fbleau()` returns a dictionary, containing:
- *min-estimate*: the minimum Bayes risk estimate (should be the one used)
- *last-estimate*: the estimate computed with the full training data
- *random-guessing*: an estimate of the random guessing error (~baseline, see [2])
- *estimates*: (if `log_errors=true`) a vector containing the value of the estimate at every step
- *min-individual-errors*: (if `log_individual_errors=true`) a vector containing
  the individual errors (`true` if error, `false` otherwise) for each test object, corresponding to
  the best (i.e., smallest) estimate

**Important note** by using `min-estimate`, you will obtain a positively
biased estimator; that is, you will be evaluating a slightly stronger adversary.
This is generally good when evaluating the robustness of defences.
However, if you do not want a biased estimator, you should use
`last-estimate`, which corresponds to training the UC method on the
entire training set.

## Example

Simple example, using the example data provided in `examples/`.

```python
import fbleau
import numpy as np

train_data = np.loadtxt("examples/geometric-4.train.csv", delimiter=",")
validation_data = np.loadtxt("examples/geometric-4.test.csv", delimiter=",")

train_secrets = train_data[:,0].astype(np.uint64)
validation_secrets = validation_data[:,0].astype(np.uint64)

train_observations = train_data[:,1:]
validation_observations = validation_data[:,1:]


fbleau.run_fbleau(train_observations, train_secrets, validation_observations,
                  validation_secrets, estimate='knn',
                  knn_strategy='ln', distance='euclidean', log_errors=False,
                  log_individual_errors=False, delta=None, qstop=None,
                  absolute=False, scale=False)
```

The above returns:

```
{'min-estimate': 0.4728,                                                        
 'last-estimate': 0.47335,                 
 'random-guessing': 0.913,                                         
 'estimates': [],     
 'min-individual-errors': []}
 ```

# TODO

Currently, the code provided here:
- is based on frequentist and nearest neighbor methods; in the future we hope
  to extend this to other ML methods; note that this does not affect the
  generality of the results, which hold for any "universally consistent" classifier,
- computes one estimate at the time (i.e., to compute multiple estimates one
  needs to run `fbleau` several times); this can change in the future.

### Short term

- [x] return various leakage measures (instead of just R*)
- [ ] resubstitution estimate
- [ ] continuous input (k-NN regressor)

### Mid term

- [ ] predictions for multiple estimators at the same time
- [ ] get training data from standard input (on-line mode)

### Maybe

- [ ] other ML methods (e.g., SVM, neural network)
- [x] Python bindings


# Hacking

If you want to play with this code, you can compile it (after
cloning the repo) with:

```
cargo build
```

To compile the Python module, you need to enable the optional
feature `python-module`; this requires nightly Rust.
Install maturin (`pip install maturin`), and then compile with:

```
maturin build --cargo-extra-args="--features python-module"
```


# References

[1] 2017, "Bayes, not Naïve: Security Bounds on Website Fingerprinting Defenses". _Giovanni Cherubin_

[2] 2018, "F-BLEAU: Fast Black-Box Leakage Estimation". _Giovanni Cherubin, Konstantinos Chatzikokolakis, Catuscia Palamidessi_.

[3] (Blog) "Machine Learning methods for Quantifying the Security of Black-boxes". https://giocher.com/pages/bayes.html

[4] (PhD thesis) "Black-box SecurityMeasuring Black-box Information Leakage via Machine Learning". https://pure.royalholloway.ac.uk/portal/files/33806285/thesis_final_after_amendments.pdf

[5] "Statistical measurement of information leakage", 2010. Konstantinos Chatzikokolakis, Tom Chothia, Apratim Guha. 
