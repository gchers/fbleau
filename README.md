# F-BLEAU

F-BLEAU is a tool for estimating the leakage of a system about its secrets in
a black-box manner (i.e., by only looking at examples of secret inputs and
respective outputs). It considers a generic system as a black-box, taking
secret inputs and returning outputs accordingly, and it measures how much the
outputs "leak" about the inputs.

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

The syntax is:
    fbleau <estimate> [options] <train> <test>

## Commands

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

## Further options

By default, `fbleau` runs until a convergence criterion is met.
We usually declare convergence if an estimate did not vary more
than `--delta`, either in relative (default) or absolute (`--abs`) value,
from its value in the last `q` examples (where `q` is specified with
`--qstop`).
One can specify more than one deltas as comma-separated values, e.g.:
`--delta=0.1,0.01,0.001`.

Optionally, one may choose to let the estimator run for all the training
set (`--run-all`), in which case `fbleau` will still report how many
examples where required for convergence.

When the system's outputs are vectors, `fbleau` by default scales their
values. The option `--no-scale` prevents this (not recommended in
general).

# Install

The code is written in `Rust`, but it is thought to be used as a
standalone command line tool.
Bindings to other programming languages (e.g., Python) may happen in the
future.

Install [rustup](https://rustup.rs) (and, consequently, `cargo`).
Then run:

```
git clone https://github.com/gchers/fbleau
cd fbleau
cargo install
```

You should now find the binary `fbleau` in your `$PATH` (if not,
check out [rustup](https://rustup.rs) again).

*Note* If `rustup` is not available on your system (e.g., \*BSD systems),
you should still be able to install `cargo` and compile `fbleau`
as shown above.

*Note* I'll also put `fbleau` on https://crates.io, hopefully soon.


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


# Authors

Giovanni Cherubin (current maintainer), Konstantinos Chatzikokolakis, Catuscia Palamidessi.

# References

[1] 2017, "Bayes, not Naïve: Security Bounds on Website Fingerprinting Defenses". _Giovanni Cherubin_

[2] 2018, "F-BLEAU: Practical Channel Leakage Estimation". _Giovanni Cherubin, Konstantinos Chatzikokolakis, Catuscia Palamidessi_.

[3] "Machine Learning methods for Quantifying the Security of Black-boxes". https://giocher.com/pages/bayes.html
