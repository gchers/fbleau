# F-BLEAU

F-BLEAU is a tool for estimating the leakage of a system in a black-box manner
(i.e., by only looking at its secret inputs and outputs). It was proposed in [2].
It considers a generic system as a black-box, taking secret inputs and returning
outputs accordingly, and it measures how much the outputs "leak" about the
inputs. This is done by only looking at observations of inputs and
respective outputs.

F-BLEAU is based on the equivalence between estimating the error of a Machine
Learning model of a specific class and the estimation of information leakage
[1,2,3].

This code was also used for the experiments of [2] on the following
evaluations: Gowalla, e-passport, and side channel attack to finite field
exponentiation.

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

# Usage

**(Section under construction)**

`fbleau` accepts as input CSV files of the following form:

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;s_1,&space;o^{(1)}_1&,&space;...,&space;o^{(d)}_1\\&space;s_2,&space;o^{(1)}_2&,&space;...,&space;o^{(d)}_2\\&space;&...&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\begin{align*}&space;s_1,&space;o^{(1)}_1&,&space;...,&space;o^{(d)}_1\\&space;s_2,&space;o^{(1)}_2&,&space;...,&space;o^{(d)}_2\\&space;&...&space;\end{align*}" title="\begin{align*} s_1, o^{(1)}_1&, ..., o^{(d)}_1\\ s_2, o^{(1)}_2&, ..., o^{(d)}_2\\ &... \end{align*}" /></a>
</p>

Each row represents an example sampled from the black-box,
where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;s_i" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;s_i" title="s_i" /></a>
is the secret input given to the system, and
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(o^{(1)}_i&,&space;...,&space;o^{(d)}_i)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;(o^{(1)}_i&,&space;...,&space;o^{(d)}_i)" title="(o^{(1)}_i&, ..., o^{(d)}_i)" /></a>
is the (vector) output of the system.
Secrets must have discrete values (although this restriction can be
lifted in the future), outputs may have either discrete or continuous
values.

## TODO

Currently, the code provided here:
- is based on frequentist and nearest neighbor methods; in the future we hope
  to extend this to other ML methods; note that this does not affect the
  generality of the results, which hold against any classifier,
- computes one estimate at the time, which can be improved in the future.

### Short term

- [x] return various leakage measures (instead of just R*)
- [ ] resubstitution estimate

### Mid term

- [ ] predictions for multiple estimators at the same time
- [ ] get training data from standard input (on-line mode)

### Maybe

- [ ] other ML methods (e.g., SVM)
- [ ] Python and Java bindings


# Authors

Giovanni Cherubin (maintainer), Konstantinos Chatzikokolakis, Catuscia Palamidessi.

# References

[1] 2017, "Bayes, not Naïve: Security Bounds on Website Fingerprinting Defenses". _Giovanni Cherubin_

[2] 2018, "F-BLEAU: Practical Channel Leakage Estimation". _Giovanni Cherubin, Konstantinos Chatzikokolakis, Catuscia Palamidessi_.

[3] "Machine Learning methods for Quantifying the Security of Black-boxes". https://giocher.com/pages/bayes.html
