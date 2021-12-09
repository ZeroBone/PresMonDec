import subprocess
import tempfile
from z3 import *


class AstReferenceWrapper:

    def __init__(self, node):
        self._node = node

    def __hash__(self):
        return self._node.hash()

    def __eq__(self, other):
        return self._node.eq(other.unwrap())

    def __repr__(self):
        return str(self._node)

    def unwrap(self):
        return self._node


def wrap_ast_ref(node):
    assert isinstance(node, AstRef)
    return AstReferenceWrapper(node)


def is_uninterpreted_variable(node):
    return is_const(node) and node.decl().kind() == Z3_OP_UNINTERPRETED


def get_formula_variables(f):

    vars_set = set()

    def ast_visitor(node):
        if is_uninterpreted_variable(node):
            vars_set.add(wrap_ast_ref(node))
        else:
            for child in node.children():
                ast_visitor(child)

    ast_visitor(f)

    return vars_set


class Z3CliError(Exception):
    pass


def timeout_ms_to_s(timeout_ms: int):
    if timeout_ms % 1000 == 0:
        return timeout_ms // 1000
    return (timeout_ms // 1000) + 1


def run_z3_cli(smt_string, timeout_ms):

    timeout_s = timeout_ms_to_s(timeout_ms)

    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w")

    try:
        smt_path = tmp.name

        tmp.write(smt_string)
        tmp.flush()

        result = subprocess.run(["z3", "-T:%d" % timeout_s,
                                 "-t:%d" % timeout_ms, "--", smt_path], capture_output=True)
    finally:
        tmp.close()
        os.unlink(tmp.name)

    if result is None:
        raise Z3CliError("Failed to run z3 as subprocess")

    result = result.stdout.decode("utf-8").rstrip()

    if result.startswith("unknown") or result.startswith("timeout"):
        return unknown

    if result.startswith("sat"):
        return sat

    if result.startswith("unsat"):
        return unsat

    raise Z3CliError("unknown z3 output: %s" % result)
