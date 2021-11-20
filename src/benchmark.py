from pathlib import Path
import time
from z3 import *
from presmondec import monadic_decomposable, monadic_decomposable_without_bound, compute_bound
from utils import get_formula_variables

# import numpy as np
# import scipy as sp
# import matplotlib
# import matplotlib.pyplot as plt


def benchmark_smts():

    base_path = Path(__file__).parent
    benchmark_root = (base_path / "../benchmark").resolve()

    for root, dirs, files in os.walk(benchmark_root):
        for file in files:
            full_file_path = os.path.join(root, file)
            assert os.path.isfile(full_file_path)
            # print("Path:", full_file_path)
            yield parse_smt2_file(full_file_path)


def run_benchmark():

    for smt in benchmark_smts():

        print("Starting testing next formula...")

        phi = And([f for f in smt])
        phi_vars = [var.unwrap() for var in get_formula_variables(phi)]

        for phi_var in phi_vars:
            b = compute_bound(phi)
            print("Bound:", b)

            start_nanos = time.perf_counter_ns()
            dec = monadic_decomposable(phi, phi_var, b)
            end_nanos = time.perf_counter_ns()

            print("Performance: %d ns" % (end_nanos - start_nanos))

            print("Decomposable:", dec)
            if not dec:
                dec_without_bound = monadic_decomposable_without_bound(phi, phi_var)
                assert not dec_without_bound
            else:
                dec_without_bound = monadic_decomposable_without_bound(phi, phi_var)
                assert dec_without_bound

        print("=" * 30)

        # fig = plt.figure()
        # ax = plt.axes()
        # x = np.linspace(0, 5, 100)
        # plt.plot(x, np.sin(x), color='Indigo', linestyle='--', linewidth=3)
        # plt.grid(visible=True, color='aqua', alpha=0.3, linestyle='-.', linewidth=2)
        # plt.show()

        time.sleep(1)


if __name__ == "__main__":
    run_benchmark()
