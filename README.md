# F-BLEAU <br><img src="https://github.com/gchers/fbleau/raw/gh-pages/img/forest.png" width="150" height="150" />
[![Build Status](https://travis-ci.org/gchers/fbleau.svg?branch=master)](https://travis-ci.org/gchers/fbleau) ![Version](https://img.shields.io/crates/v/fbleau.svg)

F-BLEAU is a tool for estimating the leakage of a system about its secrets in
a black-box manner (i.e., by only looking at examples of secret inputs and
respective outputs). It considers a generic system as a black-box, taking
secret inputs and returning outputs accordingly, and it measures how much the
outputs "leak" about the inputs. It was proposed in [2].

F-BLEAU is based on the equivalence between estimating the error of a Machine
Learning model of a specific class and the estimation of information leakage
[1,2,3].

This code was also used for the experiments of [2] on the following
evaluations: Gowalla, e-passport, and side channel attack to finite field
exponentiation.

# Getting started

F-BLEAU takes as input CSV data containing examples of system's inputs
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

    fbleau <estimate> [options] <train> <test>

## Estimates

Currently available estimates:

**log** k-NN estimate, with `k = ln(n)`, where `n` is the number of training
examples.

**log 10** k-NN estimate, with `k = log10(n)`, where `n` is the number of
training examples.

**frequentist** (or "lookup table") Standard estimate. Note that this
is only applicable when the outputs are finite; also, it does not scale
well to large systems (e.g., large input/output spaces).

Bounds and other estimates:

**nn-bound** Produces a lower bound of R* discovered by Cover and Hard ('67),
which is based on the error of the NN classifier (1-NN).

**--knn** Runs the k-NN classifier for a fixed k to be specified.
Note that this _does not_ guarantee convergence to the Bayes risk.

## Example

This example considers 100K observations generated according to a
Geometric distribution with privacy level `nu=4` (see [2] for details);
the true value of the Bayes risk is `R*=0.456` computed analytically.
The observations are split into training (80%) and test sets
(`examples/geometric-4.train.csv` and `examples/geometric-4.test.csv`
respectively).

One can run `fbleau` to compute the `log` estimate as follows:

```console
$ fbleau log examples/geometric-4.train.csv examples/geometric-4.test.csv
mapped vectors into 191 unique IDs
mapped vectors into 191 unique IDs
Random guessing error: 0.913
Estimating leakage measures...

Final estimate: 0.47475
Multiplicative Leakage: 6.037356321839082
Additive Leakage: 0.43825000000000003
Bayes security measure: 0.5199890470974808
Min-entropy Leakage: 2.593916950824318

Minimum estimate: 0.47355
Multiplicative Leakage: 6.051149425287359
Additive Leakage: 0.43945
Bayes security measure: 0.5186746987951807
Min-entropy Leakage: 2.5972092105949125
```

NOTE: depending on your machine's specs this may take a while.
`fbleau` is designed to effectively exploit many CPUs, albeit with low RAM requirements;
further optimisations are in the works.

One should look at the `Minimum estimate` (i.e., the minimum value that
the Bayes risk estimate took as the size of the training examples increases),
rather than at the `Final estimate`: indeed, the estimates do not guarantee
a monotonically decreasing behaviour.

For the `frequentist` estimate:

```console
$ fbleau frequentist examples/geometric-4.train.csv examples/geometric-4.test.csv
mapped vectors into 191 unique IDs
mapped vectors into 191 unique IDs
Random guessing error: 0.913
Estimating leakage measures...

Final estimate: 0.5621
Multiplicative Leakage: 5.033333333333335
Additive Leakage: 0.3509
Bayes security measure: 0.6156626506024097
Min-entropy Leakage: 2.3315141437165607

Minimum estimate: 0.56205
Multiplicative Leakage: 5.033908045977013
Additive Leakage: 0.35095
Bayes security measure: 0.6156078860898139
Min-entropy Leakage: 2.3316788631368333
```

## Further options

It is often useful to know the value of an estimate at every step
(i.e., for training size 1, 2, ...).
`fbleau` can output this into a file specified by `--verbose=<logfile>`.

By default, `fbleau` runs for all training data.
However, one can specify a stopping criterion, in the form of a
(delta, q)-convergence: `fbleau` stops when the estimate's value has
not changed more than delta (`--delta`), either in relative (default) or
absolute (`--absolute`) sense, for at least q steps (`--qstop`).

`fbleau` can scale the individual values of the system's output ("features")
in the `[0,1]` interval by specifying the `--scale` flag.

By default, `fbleau` uses a number of threads equal to the number of CPUs.
To limit this number, you can use `--nprocs`.

# Install

The code is written in `Rust`, but it is thought to be used as a
standalone command line tool.
Bindings to other programming languages (e.g., Python) may happen in the
future.

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


## TODO

Currently, the code provided here:
- is based on frequentist and nearest neighbor methods; in the future we hope
  to extend this to other ML methods; note that this does not affect the
  generality of the results, which hold against any classifier,
- computes one estimate at the time (i.e., to compute multiple estimates one
  needs to run `fbleau` several times); this can change in the future.

### Short term

- [x] return various leakage measures (instead of just R*)
- [ ] resubstitution estimate

### Mid term

- [ ] predictions for multiple estimators at the same time
- [ ] get training data from standard input (on-line mode)

### Maybe

- [ ] other ML methods (e.g., SVM, neural network)
- [ ] Python and Java bindings


# References

[1] 2017, "Bayes, not Naïve: Security Bounds on Website Fingerprinting Defenses". _Giovanni Cherubin_

[2] 2018, "F-BLEAU: Practical Channel Leakage Estimation". _Giovanni Cherubin, Konstantinos Chatzikokolakis, Catuscia Palamidessi_.

[3] (Blog) "Machine Learning methods for Quantifying the Security of Black-boxes". https://giocher.com/pages/bayes.html
