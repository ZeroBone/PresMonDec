import bisect
import os.path
import subprocess
import time
import logging
from pathlib import Path

import numpy as np
from z3 import *

from presmondec import monadic_decomposable, monadic_decomposable_without_bound, compute_bound, MonDecTestFailed
from utils import get_formula_variables, Z3CliError, timeout_ms_to_s

logger = logging.getLogger("premondec_benchmark")
logger.setLevel(logging.DEBUG)


def _resolve_benchmark_root():
    base_path = Path(__file__).parent
    return (base_path / "../benchmark").resolve()


def resolve_benchmark_result_root():
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


def benchmark_smts(file_size_limit: int = 0, z3_sat_check_timeout_ms: int = 0):
    assert file_size_limit >= 0

    for root, dirs, files in os.walk(_resolve_benchmark_root()):
        for file in files:
            full_file_path = os.path.join(root, file)

            if not os.path.isfile(full_file_path):
                logger.warning("File '%s' could not be found, was it deleted?", full_file_path)
                continue

            if not full_file_path.endswith(".smt2"):
                logger.warning("Cannot handle file '%s' due to unknown extention.", full_file_path)
                continue

            if file_size_limit != 0 and os.path.getsize(full_file_path) > file_size_limit:
                continue

            if z3_sat_check_timeout_ms > 0:

                timeout_s = timeout_ms_to_s(z3_sat_check_timeout_ms)

                result = subprocess.run(["z3", "-T:%d" % timeout_s,
                                         "-t:%d" % z3_sat_check_timeout_ms, "--", full_file_path], capture_output=True)

                if result.returncode != 0:
                    logger.warning("z3 terminated with nonzero exit code %d", result.returncode)

                result = result.stdout.decode("utf-8").rstrip()

                if result.startswith("unknown") or result.startswith("timeout"):
                    logger.warning("z3 has failed to solve the problem in '%s' within %d ms, "
                                   "ignoring this instance.", full_file_path, z3_sat_check_timeout_ms)
                    continue

                if not result.startswith("sat") and not result.startswith("unsat"):
                    logger.error("unknown z3 output: %s", result)
                    continue

            try:
                yield parse_smt2_file(full_file_path), full_file_path
            except Z3Exception:
                logger.warning("Could not parse benchmark file '%s'", full_file_path)


class AverageValueTracker:

    def __init__(self):
        self._x_axis = []
        self._y_running_average_values = []

    def add_point(self, x, y):
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
        self._y_running_average_values[x_axis_index] = v + float(y), 1.0 if n < 0 else n + 1.0

    def save_as_npz(self, file):
        x_axis = np.array(self._x_axis)
        y_axis = np.array([v / n for v, n in self._y_running_average_values])

        np.savez(file, x=x_axis, y=y_axis)


class BenchmarkContext:

    def __init__(self, iter_limit=0):
        self._iter_limit = iter_limit
        self._iter_number = 0

        self._cur_phi = None
        self._cur_phi_var_count = None
        self._cur_smt_path = None
        self._cur_smt_file_size = None
        self._cur_phi_var = None

        self._inconsistencies = 0
        self._md_counter = 0
        self._md_wb_counter = 0
        self._md_fail_counter = 0
        self._md_wb_fail_counter = 0

        self._var_count_bound = AverageValueTracker()

        self._md_wb_var_count = AverageValueTracker()
        self._md_wb_var_count_r = AverageValueTracker()

        self._md_wb_file_size = AverageValueTracker()
        self._md_wb_file_size_r = AverageValueTracker()

        self._md_var_count = AverageValueTracker()
        self._md_var_count_r = AverageValueTracker()

        self._md_file_size = AverageValueTracker()
        self._md_file_size_r = AverageValueTracker()

        self._bound_log_count_until_inconsistent = AverageValueTracker()
        self._bound_log_count_until_inconsistent_r = AverageValueTracker()

    def update_state(self, cur_phi=None, cur_phi_var_count=None, smt_path=None, cur_phi_var=None):

        if cur_phi is not None:
            self._cur_phi = cur_phi

        if cur_phi_var_count is not None:
            self._cur_phi_var_count = cur_phi_var_count

        if smt_path is not None:
            self._cur_smt_path = smt_path
            self._cur_smt_file_size = os.path.getsize(smt_path)

        if cur_phi_var is not None:
            self._cur_phi_var = cur_phi_var

    def _assert_formula_state_defined(self):
        assert self._cur_phi is not None
        assert self._cur_phi_var_count is not None
        assert self._cur_smt_path is not None
        assert self._cur_smt_file_size is not None

    def _assert_formula_variable_state_defined(self):
        self._assert_formula_state_defined()
        assert self._cur_phi_var is not None

    def report_bound(self, bound_bit_length: int):
        self._assert_formula_state_defined()

        self._var_count_bound.add_point(self._cur_phi_var_count, bound_bit_length)

    def report_md_perf(self, nanos, without_bound: bool):

        self._assert_formula_variable_state_defined()

        if nanos is None:

            if without_bound:
                self._md_wb_fail_counter += 1
            else:
                self._md_fail_counter += 1

            return

        ms = nanos // 1000000

        md_var_count, md_var_count_r, md_file_size, md_file_size_r = (
            self._md_wb_var_count,
            self._md_wb_var_count_r,
            self._md_wb_file_size,
            self._md_wb_file_size_r) if without_bound else (
            self._md_var_count,
            self._md_var_count_r,
            self._md_file_size,
            self._md_file_size_r
        )

        md_var_count.add_point(ms, self._cur_phi_var_count)
        md_var_count_r.add_point(self._cur_phi_var_count, ms)

        md_file_size.add_point(ms, self._cur_smt_file_size)
        md_file_size_r.add_point(self._cur_smt_file_size, ms)

        if without_bound:
            self._md_wb_counter += 1
        else:
            self._md_counter += 1

    def report_md_log_count_until_inconsistent(self, log_count: int, initial_bound_bit_length: int):

        self._assert_formula_variable_state_defined()

        self._bound_log_count_until_inconsistent.add_point(initial_bound_bit_length, log_count)
        self._bound_log_count_until_inconsistent_r.add_point(log_count, initial_bound_bit_length)

    def report_mondec_results(self, monadic_decomposable, monadic_decomposable_without_bound):
        self._assert_formula_variable_state_defined()

        if monadic_decomposable != monadic_decomposable_without_bound:
            self._inconsistencies += 1
            logger.error("Inconsistency in '%s' on variable '%s'", self._cur_smt_path, self._cur_phi_var)

    def log_stats(self):
        logger.info("Inconsistencies so far: %5d", self._inconsistencies)
        logger.info("Benchmark runs: With bound: %d Without bound: %d",
                    self._md_counter, self._md_wb_counter)
        logger.info("Benchmark failed runs: With bound: %d Without bound: %d",
                    self._md_fail_counter, self._md_wb_fail_counter)
        logger.info("Benchmark total runs: With bound: %d Without bound: %d",
                    self._md_counter + self._md_fail_counter,
                    self._md_wb_counter + self._md_wb_fail_counter)

    def next_round(self) -> bool:

        self._iter_number += 1

        if self._iter_limit != 0 and self._iter_number == self._iter_limit:
            return True

        assert self._iter_limit == 0 or self._iter_number < self._iter_limit

        if self._iter_number % 10 == 0:
            self.log_stats()
            self.export_data()

        return False

    def export_data(self):

        output = resolve_benchmark_result_root()

        prefix = "%05d_" % self._iter_number

        for avt, file_name in [
            (self._var_count_bound, "var_count_bound.npz"),
            (self._md_wb_var_count, "md_wb_var_count.npz"),
            (self._md_wb_var_count_r, "md_wb_var_count_r.npz"),
            (self._md_wb_file_size, "md_wb_file_size.npz"),
            (self._md_wb_file_size_r, "md_wb_file_size_r.npz"),
            (self._md_var_count, "md_var_count.npz"),
            (self._md_var_count_r, "md_var_count_r.npz"),
            (self._md_file_size, "md_file_size.npz"),
            (self._md_file_size_r, "md_file_size_r.npz"),
            (self._bound_log_count_until_inconsistent, "bound_log_count_until_inc.npz"),
            (self._bound_log_count_until_inconsistent_r, "bound_log_count_until_inc_r.npz")
        ]:
            avt.save_as_npz(os.path.join(output, prefix + file_name))

    def get_iteration_number(self) -> int:
        return self._iter_number


def run_benchmark(iter_limit=0, vars_per_formula_limit=5,
                  z3_sat_check_timeout_ms=0, file_size_limit=0, z3_timeout_ms=0):
    ctx = BenchmarkContext(iter_limit)

    logger.info("Benchmark started.")

    for smt, smt_path in benchmark_smts(file_size_limit, z3_sat_check_timeout_ms):

        logger.info("Iteration: %06d Considering '%s'", ctx.get_iteration_number(), smt_path)

        try:
            phi = And([f for f in smt])
            phi_vars = [var.unwrap() for var in get_formula_variables(phi)]
            var_count = len(phi_vars)

            phi = And([phi] + [v >= 0 for v in phi_vars])
        except TypeError:
            logger.warning("Could not cleanup formula, ignoring it - is it a valid presburger arithmetic formula?")
            continue

        start_nanos = time.perf_counter_ns()
        b = compute_bound(phi)
        end_nanos = time.perf_counter_ns()

        bound_b_computation_time = end_nanos - start_nanos

        b_bit_length = b.bit_length()

        ctx.update_state(phi, var_count, smt_path)

        ctx.report_bound(b_bit_length)

        for phi_var in phi_vars[:vars_per_formula_limit]:

            logger.info("Decomposing on variable '%s'", phi_var)

            ctx.update_state(cur_phi_var=phi_var)

            try:
                start_nanos = time.perf_counter_ns()
                dec = monadic_decomposable(phi, phi_var, b, z3_timeout_ms)
                end_nanos = time.perf_counter_ns()

                ctx.report_md_perf(
                    bound_b_computation_time + end_nanos - start_nanos,
                    False
                )
            except Z3CliError as e:
                logger.error("z3 cli error: %s", str(e))
                continue
            except MonDecTestFailed:
                ctx.report_md_perf(None, False)
                continue

            logger.info("Monadically decomposable (with bound): %s", "yes" if dec else "no")

            smaller_bound = b.bit_length()
            log_count = 0

            while True:

                try:
                    dec_with_smaller_bound = monadic_decomposable(phi, phi_var, smaller_bound, z3_timeout_ms)
                except Z3CliError as e:
                    logger.error("z3 cli error: %s", str(e))
                    break
                except MonDecTestFailed:
                    break

                if smaller_bound <= 2 or dec != dec_with_smaller_bound:
                    # either we have reached the bound such that the log doesn't decrease anymore
                    # or the monadic decomposition results aren't consistent anymore

                    ctx.report_md_log_count_until_inconsistent(log_count, b_bit_length)

                    break

                smaller_bound = smaller_bound.bit_length()
                log_count += 1

            logger.info("max{k:decomposition with bound log^k(B) is consistent} = %d", log_count)

            try:
                start_nanos = time.perf_counter_ns()
                dec_without_bound = monadic_decomposable_without_bound(phi, phi_var, timeout_ms=z3_timeout_ms)
                end_nanos = time.perf_counter_ns()

                ctx.report_md_perf(end_nanos - start_nanos, True)
            except Z3CliError as e:
                logger.error("z3 cli error: %s", str(e))
                continue
            except MonDecTestFailed:
                logger.info("Could not decompose on '%s' without bound", phi_var)
                ctx.report_md_perf(None, True)
                continue

            ctx.report_mondec_results(dec, dec_without_bound)

            logger.info("Iteration for variable '%s' completed without monadic decomposition failures", phi_var)

        if ctx.next_round():
            # round limit reached
            break

    logger.info("Benchmarking complete!")

    return ctx


if __name__ == "__main__":

    fh = logging.FileHandler(os.path.join(resolve_benchmark_result_root(), "benchmark.log"))
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s %(levelname)7s]: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    if "--clean" in sys.argv[1:]:
        remove_unparsable_benchmarks()
    elif len(sys.argv) < 3:
        print("[ITERATION_LIMIT] [VARS_PER_FORMULA_LIMIT] arguments missing")
        print("Usage: python benchmark.py [ITERATION_LIMIT] [VARS_PER_FORMULA_LIMIT]\n\t"
              "[SAT_CHECK_TIMEOUT_MS] [Z3_TIMEOUT_MS] [FILE_SIZE_LIMIT_MiB]")
        print("The last two arguments are optional.")
    else:

        iter_limit = int(sys.argv[1])
        vars_per_formula_limit = int(sys.argv[2])

        assert iter_limit >= 0
        assert vars_per_formula_limit >= 1

        z3_sat_check_timeout_ms = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
        assert z3_sat_check_timeout_ms >= 0

        z3_timeout_ms = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
        assert z3_timeout_ms >= 0

        file_size_limit_kb = int(sys.argv[5]) if len(sys.argv) >= 6 else 0
        assert file_size_limit_kb >= 0

        logger.info("\n"
                    "______             ___  ___           ______          \n"
                    "| ___ \\            |  \\/  |           |  _  \\         \n"
                    "| |_/ / __ ___  ___| .  . | ___  _ __ | | | |___  ___ \n"
                    "|  __/ '__/ _ \\/ __| |\\/| |/ _ \\| '_ \\| | | / _ \\/ __|\n"
                    "| |  | | |  __/\\__ \\ |  | | (_) | | | | |/ /  __/ (__ \n"
                    "\\_|  |_|  \\___||___|_|  |_/\\___/|_| |_|___/ \\___|\\___|")

        logger.info("Iteration limit: %d", iter_limit)
        logger.info("Maximum variables per formula limit: %d", vars_per_formula_limit)

        if z3_sat_check_timeout_ms != 0:
            logger.info("Sat check: enabled, timeout = %d ms", z3_sat_check_timeout_ms)
        else:
            logger.info("Sat check: disabled")

        if z3_timeout_ms != 0:
            logger.info("Z3 timeout: enabled, timeout = %d ms", z3_timeout_ms)
        else:
            logger.info("Z3 timeout: disabled")

        if file_size_limit_kb != 0:
            logger.info("File size limit: enabled, limit = %d KB", file_size_limit_kb)
        else:
            logger.info("File size limit: disabled")

        ctx = run_benchmark(
            iter_limit,
            vars_per_formula_limit,
            z3_sat_check_timeout_ms,
            file_size_limit_kb * 1000,
            z3_timeout_ms
        )

        ctx.log_stats()
        ctx.export_data()
