# PresMonDec

PresMonDec is at tool that checks whether a quantifier-free Formula in Presburger arithmetic is monadically decomposable. Congruence relations are supported.

# Installation

Following dependencies have to be installed in order to be able to use all features of this project:
* [Z3](https://github.com/Z3Prover/z3)
* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)

Only `z3` is however mandatory for the decomposition checkers.

To install `z3`, follow the official tutorial for your operating system.
Please note that `z3` should be available as a CLI service for the monadic decomposition to work when the timeout is specified explicitly (see the [Usage](#Usage) section below).
`numpy` and `matplotlib` can be installed as follows:
```shell
pip install numpy
pip install matplotlib
```

# Usage

Two different decomposability checking methods for Presburger arithmetic (over natural numbers) are implemented. Both of them can be used by importing the `src/presmondec.py` module.

The decomposability checkers support modular arithmetic, i.e. congruence relations like `5 * x ≡ 3 * y mod 7`. These relations must however be constructed using the `congruent` function defined in the same module.
The arguments are the left and right-hand sides as well as the modulo which must be a constant. So, for example, the above statement must be constructed using
```python
from z3 import *
from presmondec import congruent

x, y = Ints("x y z")
expr = congruent(5 * x, 3 * y, 7)
```
Also, the monadic decomposition may not work if the formula has satisfying assignments where some variable values are negative.
It is therefore required to add `x >= 0` constraints for all variables `x` if the formula doesn't already restrict negative values for variables. 

## First method

The first method is based on the [Monadic Decomposition in Integer Linear Arithmetic](https://arxiv.org/abs/2004.12371) paper by Matthew Hague, Anthony Widjaja Lin, Philipp Rümmer and Zhilin Wu.
It is important that there, a bound `B` for essentially the maximum value the formula can "address explicitly" is computed before constructing a formula for `z3` that expresses the fact that the formula given is monadically decomposable.

To use this method, call the `monadic_decomposable` function from the `src/presmondec.py` module.
The arguments are:
1. Formula to be tested
2. Variable that we want to test the monadic decomposition on
3. *Optional*: The bound `B` in case it is already known (it can be computed using the `compute_bound` function)
4. *Optional*: Timeout (in milliseconds) after which the decomposition shall be aborted and the `MonDecTestFailed` exception shall be raised

### Example

```python
from z3 import *
from presmondec import monadic_decomposable, congruent

x, y, z = Ints("x y z")

phi = And([
    x >= 0,
    y >= 0,
    z >= 0,
    x + 2 * y >= 5,
    z < 5,
    congruent(x, y, 2)
])

# will return True because phi is monadically decomposable on x
dec = monadic_decomposable(phi, x)

print("Monadically decomposable on x:", dec)
```

## Second method

In the second method the monadic decomposition is tested by constructing a formula that essentially describes the existence of a bound `B` with an existential quantifier.
In accordance with Proposition 2.3 in the paper above, such a bound exists and every element is equivalent to some element bounded by this bound if and only if the formula is monadically decomposable.

This method is implemented as a function `monadic_decomposable_without_bound` in `src/presmondec.py`. The arguments are:
1. Formula to be tested
2. Variable that we want to test the monadic decomposition on
3. *Optional*: An upper bound for the maximum of all minimal representatives of equivalence classes, where the equivalence relation is described in Proposition 2.3 in the paper above.
4. *Optional*: Timeout (in milliseconds) after which the decomposition shall be aborted and the `MonDecTestFailed` exception shall be raised

### Example

```python
from z3 import *
from presmondec import monadic_decomposable_without_bound

x, y, z = Ints("x y z")

phi = And(Or(x == y, y == z), x >= y, y >= z, z >= 0)

# will return False because phi is not monadically decomposable on x
dec = monadic_decomposable_without_bound(phi, x)

print("Monadically decomposable on x:", dec)
```

# Benchmark

Both monadic decomposability checking methods can be tested on a big dataset of formulas (located in the `benchmark` directory) by running
```
python benchmark.py
    [ITERATION_LIMIT]
    [VARS_PER_FORMULA_LIMIT]
    [SAT_CHECK_TIMEOUT_MS]
    [Z3_TIMEOUT_MS]
    [FILE_SIZE_LIMIT_KB]
```
where the five command-line arguments mean the following:
1. `[ITERATION_LIMIT]` is the maximum amount of formulas which should be successfully considered by the benchmark, where "successfully considered" means that both monadic decomposition methods succeed for the current formula. Set this parameter to `0` to allow any amount of iterations.
2. `[VARS_PER_FORMULA_LIMIT]` is the maximum amount of variables per formula on which the formula should be monadically decomposed (with both methods). 
3. *Optional*: `[SAT_CHECK_TIMEOUT_MS]`: if some formula cannot be solved by `z3` within this amount of milliseconds, then the formula gets ignored by the benchmark. Set this parameter to `0` to disable prior running of `z3` to determine how fast satisfiability of a formula can be determined. **Default value**: `0`
4. *Optional*: `[Z3_TIMEOUT_MS]`: If a `z3` call initiated from inside a monadic decomposition check fails to produce an answer within the amount of milliseconds specified by this argument, `z3` is aborted. Set this parameter to `0` to disable the timeout. **Default value**: `0`
5. *Optional*: `[FILE_SIZE_LIMIT_KB]` is the maximum `.smt2` file size in kilobytes, any files exceeding this limit will be ignored by the benchmark. If this parameter is set to `0`, the benchmark will consider all files regardless of their size. **Default value**: `0`   

## Results

![smt2 file size and average mondec performance comparison](benchmark_results/md_file_size_r.png)
![average mondec performance and smt2 file size comparison](benchmark_results/md_file_size.png)

![variable count and average mondec performance comparison](benchmark_results/md_var_count_r.png)
![average mondec performance and variable count comparison](benchmark_results/md_var_count.png)

![variable count and bitlength of bound comparison](benchmark_results/var_count_bound_simple.png)
![variable count and bitlength of bound comparison](benchmark_results/var_count_bound.png)

![analysis of how efficient the bound is](benchmark_results/log_count_until_inc_dist.png)
![analysis of how efficient the bound is](benchmark_results/bound_log_count_until_inc_r.png)