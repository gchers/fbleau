# F-BLEAU

F-BLEAU is a tool for estimating the leakage of a system in a black-box manner
(i.e., by only looking at its secret inputs and outputs) we proposed in [2].

This tool is based on the equivalence between Machine Learning and black-box
estimation of information leakage [1,2,3].

This is also the code used for the experiments of [2] on the following
evaluations: Gowalla, e-passport, and side channel attack to finite field
exponentiation.


## Status

For now, this tool:
- only estimates the Bayes risk (one has to derive the desired leakage measure
  from it manually),
- is based on frequentist and nearest neighbor methods; in the future we hope
  to extend this to other ML methods,
- computes one estimate at the time; it is possible however to compute
  many estimates at the same time.


## Compile

Get yourself `cargo` (maybe using `rustup`).

Run:

```
git clone <this repo>
cd fbleau
cargo install
```

Now you can run the binary with `fbleau`.

## Running

`fbleau` accepts .csv files (for training and test data) of the following
form:

    y_1, x^1_1, x^2_1, ..., x^d_1
    y_2, x^1_2, x^2_2, ..., x^d_2
    ...

where $y_i$ is the label of the `i`-th object, and the object itself is
a vector of `d` real numbers: `(x^1_i, x^2_i, ..., x^d_i)`.

## TODO

### Short term

[ ] return various leakage measures (instead of R*)
[ ] resubstitution estimate

### Mid term

[ ] predictions for multiple estimators at the same time
[ ] get training data from standard input (on-line mode)

### Maybe

[ ] other ML methods (e.g., SVM)
[ ] Python and Java bindings


## Authors

Giovanni Cherubin (maintainer), Konstantinos Chatzikokolakis, Catuscia Palamidessi.

## References

[1] 2017, "Bayes, not Naïve: Security Bounds on Website Fingerprinting Defenses" _Giovanni Cherubin_
[2] 2018, "F-BLEAU: Practical Channel Leakage Estimation" _Giovanni Cherubin, Konstantinos Chatzikokolakis, Catuscia Palamidessi_.
[3] "Machine Learning methods for Quantifying the Security of Black-boxes", https://giocher.com/pages/bayes.html
