from pathlib import Path
import time
from z3 import *
from presmondec import monadic_decomposable, monadic_decomposable_without_bound, compute_bound, MonDecTestFailed
from utils import get_formula_variables

import numpy as np
import matplotlib.pyplot as plt


def benchmark_smts():

    base_path = Path(__file__).parent
    benchmark_root = (base_path / "../benchmark").resolve()

    for root, dirs, files in os.walk(benchmark_root):
        for file in files:
            full_file_path = os.path.join(root, file)
            assert os.path.isfile(full_file_path)
            # print("Path:", full_file_path)
            yield parse_smt2_file(full_file_path), full_file_path


class BenchmarkContext:

    def __init__(self, rounds_limit=0):
        self.rounds_limit = rounds_limit
        self.var_count_to_stats = {}
        self.round_no = 0
        self.cur_phi = None
        self.cur_phi_var_count = None
        self.cur_smt_path = None
        self.cur_phi_var = None
        self.inconsistencies = []

    def update_state(self, cur_phi=None, cur_phi_var_count=None, smt_path=None, cur_phi_var=None):

        if cur_phi is not None:
            self.cur_phi = cur_phi

        if cur_phi_var_count is not None:
            self.cur_phi_var_count = cur_phi_var_count

        if smt_path is not None:
            self.cur_smt_path = smt_path

        if cur_phi_var is not None:
            self.cur_phi_var = cur_phi_var

    def _assert_formula_state_defined(self):
        assert self.cur_phi is not None
        assert self.cur_phi_var_count is not None
        assert self.cur_smt_path is not None

    def _assert_formula_variable_state_defined(self):
        self._assert_formula_state_defined()
        assert self.cur_phi_var is not None

    def report_bound(self, bound):
        self._assert_formula_state_defined()

    def report_monadic_decomposable_without_bound_perf(self, nanos):

        self._assert_formula_variable_state_defined()

        if nanos is None:
            pass

        ms = nanos / 1e6

    def report_monadic_decomposable_perf(self, nanos):

        self._assert_formula_variable_state_defined()

        if nanos is None:
            pass

        ms = nanos / 1e6

    def report_mondec_results(self, monadic_decomposable, monadic_decomposable_without_bound):
        self._assert_formula_variable_state_defined()

        if monadic_decomposable != monadic_decomposable_without_bound:
            self.inconsistencies.append((self.cur_smt_path, self.cur_phi_var, self.cur_phi))

    def next_round(self) -> bool:

        self.round_no += 1

        if self.rounds_limit != 0 and self.round_no == self.rounds_limit:
            return True

        assert self.round_no < self.rounds_limit

        if True or self.round_no % 20 == 0:
            print("Inconsistencies so far: %5d" % len(self.inconsistencies))
            print("Starting round %5d..." % self.round_no)

        return False

    def export_graphs(self):

        fig, ax = plt.subplots()

        x = np.linspace(0, 10, 30)
        y = np.cos(x)

        ax.plot(x, y, 'o', color='black')

        fig.tight_layout()
        fig.savefig("test.svg")

        plt.show()


def run_benchmark(rounds_limit=0):

    ctx = BenchmarkContext(rounds_limit)

    print("Benchmark started.")

    for smt, smt_path in benchmark_smts():

        phi = And([f for f in smt])
        phi_vars = [var.unwrap() for var in get_formula_variables(phi)]
        var_count = len(phi_vars)

        b = compute_bound(phi)

        ctx.update_state(phi, var_count, smt_path)

        ctx.report_bound(b)

        for phi_var in phi_vars:

            ctx.update_state(cur_phi_var=phi_var)

            try:
                start_nanos = time.perf_counter_ns()
                dec = monadic_decomposable(phi, phi_var, b)
                end_nanos = time.perf_counter_ns()

                ctx.report_monadic_decomposable_perf(end_nanos - start_nanos)
            except MonDecTestFailed:
                ctx.report_monadic_decomposable_perf(None)
                continue

            try:
                start_nanos = time.perf_counter_ns()
                dec_without_bound = monadic_decomposable_without_bound(phi, phi_var)
                end_nanos = time.perf_counter_ns()

                ctx.report_monadic_decomposable_without_bound_perf(end_nanos - start_nanos)
            except MonDecTestFailed:
                ctx.report_monadic_decomposable_without_bound_perf(None)
                continue

            ctx.report_mondec_results(dec, dec_without_bound)

        if ctx.next_round():
            break

    print("Benchmark finished.")

    return ctx


if __name__ == "__main__":

    set_option(timeout=10 * 1000)

    run_benchmark(10).export_graphs()
