from pathlib import Path
import time
import signal
from contextlib import contextmanager
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


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):

    def signal_handler(_, __):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)


class BenchmarkContext:

    def __init__(self):
        self.var_count_to_stats = {}
        self.round_no = 0

    def report_bound(self, var_count, bound):
        pass

    def report_monadic_decomposable_without_bound_perf(self, var_count, nanos):
        ms = nanos / 1e6

    def report_monadic_decomposable_perf(self, var_count, nanos):
        ms = nanos / 1e6

    def report_mondec_results(self, monadic_decomposable, monadic_decomposable_without_bound):
        pass

    def next_round(self):
        self.round_no += 1
        if self.round_no % 20 == 0:
            print("Starting round %5d..." % self.round_no)

    def completed(self):
        print("Benchmark finished.")


def run_benchmark(dec_timeout):

    ctx = BenchmarkContext()

    for smt in benchmark_smts():

        phi = And([f for f in smt])
        phi_vars = [var.unwrap() for var in get_formula_variables(phi)]
        var_count = len(phi_vars)

        b = compute_bound(phi)

        ctx.report_bound(var_count, b)

        for phi_var in phi_vars:

            try:
                with time_limit(dec_timeout):
                    start_nanos = time.perf_counter_ns()
                    dec = monadic_decomposable(phi, phi_var, b)
                    end_nanos = time.perf_counter_ns()
            except TimeoutException as e:
                print("monadic_decomposable() timeout")
                continue

            ctx.report_monadic_decomposable_perf(var_count, end_nanos - start_nanos)

            try:
                with time_limit(dec_timeout):
                    start_nanos = time.perf_counter_ns()
                    dec_without_bound = monadic_decomposable_without_bound(phi, phi_var)
                    end_nanos = time.perf_counter_ns()
            except TimeoutException as e:
                print("monadic_decomposable_without_bound() timeout")
                continue

            ctx.report_monadic_decomposable_without_bound_perf(var_count, end_nanos - start_nanos)

            ctx.report_mondec_results(dec, dec_without_bound)

        ctx.next_round()

    return ctx.completed()


if __name__ == "__main__":
    print("Benchmark started.")
    run_benchmark(30)
