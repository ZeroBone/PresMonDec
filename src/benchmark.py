import bisect
import os.path
import subprocess
import time
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from z3 import *

from presmondec import monadic_decomposable, monadic_decomposable_without_bound, compute_bound, MonDecTestFailed
from utils import get_formula_variables

logger = logging.getLogger("premondec_benchmark")
logger.setLevel(logging.DEBUG)


def _resolve_benchmark_root():
    base_path = Path(__file__).parent
    return (base_path / "../benchmark").resolve()


def _resolve_benchmark_result_root():
    base_path = Path(__file__).parent
    return (base_path / "../benchmark_results").resolve()


def remove_unparsable_benchmarks():

    removed_count = 0
    remaining_count = 0

    for root, dirs, files in os.walk(_resolve_benchmark_root()):
        for file in files:
            full_file_path = os.path.join(root, file)

            assert os.path.isfile(full_file_path)

            try:
                parse_smt2_file(full_file_path)
            except Z3Exception:
                logger.info("Removing file '%s'", full_file_path)
                os.remove(full_file_path)
                removed_count += 1
                continue

            remaining_count += 1

    logger.info("Done - removed: %d remaining: %d", removed_count, remaining_count)


def benchmark_smts(file_size_limit: int):

    assert file_size_limit >= 0

    for root, dirs, files in os.walk(_resolve_benchmark_root()):
        for file in files:
            full_file_path = os.path.join(root, file)

            assert os.path.isfile(full_file_path)

            if file_size_limit != 0 and os.path.getsize(full_file_path) >= file_size_limit:
                continue

            try:
                yield parse_smt2_file(full_file_path), full_file_path
            except Z3Exception:
                logger.warning("Could not parse benchmark file '%s'", full_file_path)


class AverageValuePlotter:

    def __init__(self):
        self._x_axis = []
        self._y_running_average_values = []

    def add_point(self, x, y):

        y = float(y)

        # using binary search, find the appropriate place
        # for the value of x on the x axis
        x_axis_index = bisect.bisect_left(self._x_axis, x)

        if x_axis_index >= len(self._x_axis) or self._x_axis[x_axis_index] != x:
            # then x value is not yet known

            self._x_axis[x_axis_index:x_axis_index] = [x]

            # -1 is needed to prevent division by zero in case there
            # exists an x-value without a y-value
            self._y_running_average_values[x_axis_index:x_axis_index] = [(0.0, -1.0)]

        v, n = self._y_running_average_values[x_axis_index]
        self._y_running_average_values[x_axis_index] = v + y, 1.0 if n < 0 else n + 1.0

    def plot(self):
        fig, ax = plt.subplots()

        x_axis = np.array(self._x_axis)
        y_axis = np.array([v / n for v, n in self._y_running_average_values])

        ax.plot(x_axis, y_axis, 'o', color='black')

        return fig, ax


class BenchmarkContext:

    def __init__(self, iter_limit=0):
        self._iter_limit = iter_limit
        self._iter_number = 0

        self._cur_phi = None
        self._cur_phi_var_count = None
        self._cur_smt_path = None
        self._cur_phi_var = None
        self._inconsistencies = 0

        self._stat_var_count_bound = AverageValuePlotter()
        self._md_without_bound_var_count = AverageValuePlotter()
        self._md_var_count = AverageValuePlotter()

    def update_state(self, cur_phi=None, cur_phi_var_count=None, smt_path=None, cur_phi_var=None):

        if cur_phi is not None:
            self._cur_phi = cur_phi

        if cur_phi_var_count is not None:
            self._cur_phi_var_count = cur_phi_var_count

        if smt_path is not None:
            self._cur_smt_path = smt_path

        if cur_phi_var is not None:
            self._cur_phi_var = cur_phi_var

    def _assert_formula_state_defined(self):
        assert self._cur_phi is not None
        assert self._cur_phi_var_count is not None
        assert self._cur_smt_path is not None

    def _assert_formula_variable_state_defined(self):
        self._assert_formula_state_defined()
        assert self._cur_phi_var is not None

    def report_bound(self, bound):
        self._assert_formula_state_defined()

        self._stat_var_count_bound.add_point(self._cur_phi_var_count, bound.bit_length())

    def report_monadic_decomposable_without_bound_perf(self, nanos):

        self._assert_formula_variable_state_defined()

        if nanos is None:
            return

        ms = nanos / 1e6

        self._md_without_bound_var_count.add_point(ms, self._cur_phi_var_count)

    def report_monadic_decomposable_perf(self, nanos):

        self._assert_formula_variable_state_defined()

        if nanos is None:
            return

        ms = nanos / 1e6

        self._md_var_count.add_point(ms, self._cur_phi_var_count)

    def report_mondec_results(self, monadic_decomposable, monadic_decomposable_without_bound):
        self._assert_formula_variable_state_defined()

        if monadic_decomposable != monadic_decomposable_without_bound:
            self._inconsistencies += 1
            logger.error("Inconsistency in '%s' on variable '%s'", self._cur_smt_path, self._cur_phi_var)

    def next_round(self) -> bool:

        self._iter_number += 1

        if self._iter_limit != 0 and self._iter_number == self._iter_limit:
            return True

        assert self._iter_limit == 0 or self._iter_number < self._iter_limit

        if self._iter_number % 5 == 0:
            self.export_graphs()

        if self._iter_number % 10 == 0:
            logger.info("Inconsistencies so far: %5d", self._inconsistencies)
            logger.info("Starting iteration: %5d", self._iter_number)

        return False

    def export_graphs(self):

        output_path = _resolve_benchmark_result_root()

        fig, ax = self._stat_var_count_bound.plot()

        ax.set_xlabel("Variable count")
        ax.set_ylabel("Average bit length of B")

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, "%05d_bound_var_count.svg" % self._iter_number))

        fig, ax = self._md_without_bound_var_count.plot()

        ax.set_xlabel("monadic_decomposable_without_bound performance (ms)")
        ax.set_ylabel("Average variable count")

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, "%05d_monadic_decomposable_without_bound.svg" % self._iter_number))

        fig, ax = self._md_var_count.plot()

        ax.set_xlabel("monadic_decomposable performance (ms)")
        ax.set_ylabel("Average variable count")

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, "%05d_monadic_decomposable.svg" % self._iter_number))


def run_benchmark(iter_limit=0, vars_per_formula_limit=5, z3_sat_check_timeout_ms=0, file_size_limit=0):

    ctx = BenchmarkContext(iter_limit)

    logger.info("Benchmark started.")

    for smt, smt_path in benchmark_smts(file_size_limit):

        logger.info("Considering '%s'", smt_path)

        if z3_sat_check_timeout_ms > 0:

            result = subprocess.run(["z3", "-t:%d" % z3_sat_check_timeout_ms, "--", smt_path], capture_output=True)

            if result.returncode != 0:
                logger.warning("z3 has terminated with nonzero exit code %d", result.returncode)

            result = result.stdout.decode("utf-8").rstrip()

            if result.startswith("unknown"):
                logger.warning("z3 has failed to solve the problem in '%s' withing %d ms, "
                               "ignoring this instance.", smt_path, z3_sat_check_timeout_ms)
                continue

            if not result.startswith("sat") and not result.startswith("unsat"):
                logger.error("unknown z3 output: %s", result)
                continue

        phi = And([f for f in smt])
        phi_vars = [var.unwrap() for var in get_formula_variables(phi)]
        var_count = len(phi_vars)

        phi = And([phi] + [v >= 0 for v in phi_vars])

        start_nanos = time.perf_counter_ns()
        b = compute_bound(phi)
        end_nanos = time.perf_counter_ns()

        bound_b_computation_time = end_nanos - start_nanos

        ctx.update_state(phi, var_count, smt_path)

        ctx.report_bound(b)

        for phi_var in phi_vars[:vars_per_formula_limit]:

            ctx.update_state(cur_phi_var=phi_var)

            try:
                start_nanos = time.perf_counter_ns()
                dec = monadic_decomposable(phi, phi_var, b)
                end_nanos = time.perf_counter_ns()

                ctx.report_monadic_decomposable_perf(bound_b_computation_time + end_nanos - start_nanos)
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
            # round limit reached
            break

    logger.info("Benchmarking complete!")

    return ctx


if __name__ == "__main__":

    fh = logging.FileHandler(os.path.join(_resolve_benchmark_result_root(), "benchmark.log"))
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s %(levelname)7s]: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    if "--clean" in sys.argv[1:]:
        remove_unparsable_benchmarks()
    elif len(sys.argv) < 3:
        print("[ITERATION_LIMIT] [VARS_PER_FORMULA_LIMIT] arguments missing")
        print("Usage: python benchmark.py [ITERATION_LIMIT] [VARS_PER_FORMULA_LIMIT] [SAT_CHECK_TIMEOUT_MS] ["
              "FILE_SIZE_LIMIT_MiB]")
        print("The last two arguments are optional.")
    else:

        iter_limit = int(sys.argv[1])
        vars_per_formula_limit = int(sys.argv[2])

        assert iter_limit >= 0
        assert vars_per_formula_limit >= 1

        z3_sat_check_timeout_ms = int(sys.argv[3]) if len(sys.argv) >= 4 else 0

        assert z3_sat_check_timeout_ms >= 0

        file_size_limit_mib = int(sys.argv[4]) if len(sys.argv) >= 5 else 0

        assert file_size_limit_mib >= 0

        logger.info("Iteration limit: %d", iter_limit)
        logger.info("Maximum variables per formula limit: %d", vars_per_formula_limit)

        if z3_sat_check_timeout_ms != 0:
            logger.info("Sat check: enabled, timeout = %d ms", z3_sat_check_timeout_ms)
        else:
            logger.info("Sat check: disabled")

        if file_size_limit_mib != 0:
            logger.info("File size limit: enabled, limit = %d MiB", file_size_limit_mib)
        else:
            logger.info("File size limit: disabled")

        set_option(timeout=10 * 1000)

        ctx = run_benchmark(
            iter_limit,
            vars_per_formula_limit,
            z3_sat_check_timeout_ms,
            file_size_limit_mib << 20
        )

        ctx.export_graphs()
