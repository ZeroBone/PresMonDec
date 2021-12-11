from z3 import *
from utils import wrap_ast_ref, is_uninterpreted_variable, get_formula_variables, run_z3_cli, TooDeepFormulaError


class MonDecTestFailed(Exception):
    pass


def compute_bound(f) -> int:

    visited = set()

    def ast_visitor(node):

        if is_int_value(node):
            return node.as_long().bit_length(), 0, 0

        if is_uninterpreted_variable(node):
            # this is a variable
            return 0, 0, 1

        if not is_app(node):
            raise MonDecTestFailed("the AST of the formula contains a leaf that is not a integer constant")

        max_d = 1
        linear_equality_count = 0
        variable_count = 0

        for child in node.children():

            child_wrapped = wrap_ast_ref(child)

            if child_wrapped in visited:
                continue

            visited.add(child_wrapped)

            d, n, m = ast_visitor(child)

            max_d = max(max_d, d)
            linear_equality_count += n
            variable_count += m

        if node.decl().kind() == Z3_OP_EQ:
            # found an equality
            linear_equality_count += 1

        return max_d, linear_equality_count, variable_count

    visited.add(wrap_ast_ref(f))

    try:
        if f.decl().name() == "or":

            d = 1
            n = 0
            m = 0

            for child in f.children():

                child_wrapped = wrap_ast_ref(child)

                if child_wrapped in visited:
                    continue

                visited.add(child_wrapped)

                cur_d, cur_n, cur_m = ast_visitor(child)

                d = max(d, cur_d)
                n = max(n, cur_n)
                m += cur_m

        else:
            d, n, m = ast_visitor(f)
    except (RecursionError, ctypes.ArgumentError):
        raise MonDecTestFailed("recursion error: stack overflow")

    final_shift = d * n * m + 3

    if final_shift > 100000:
        raise MonDecTestFailed("exponent is galactic: %d" % (final_shift + 1))

    return 1 << final_shift


def congruent(lhs, rhs, modulo):
    return (lhs - rhs) % modulo == 0


def _same_div_transform_constraint(x_1, x_2, x, div_constraint):

    lhs, rhs, modulo = div_constraint

    lhs_x1 = substitute(lhs, (x, x_1))
    rhs_x1 = substitute(rhs, (x, x_1))

    lhs_x2 = substitute(lhs, (x, x_2))
    rhs_x2 = substitute(rhs, (x, x_2))

    return congruent(lhs_x1, rhs_x1, modulo) == congruent(lhs_x2, rhs_x2, modulo)


def _same_div(x_1, x_2, x, phi):

    visited = set()
    div_constraints = set()

    def detect_div_constraints(node):
        if node.decl().kind() == Z3_OP_EQ and \
                node.arg(0).decl().kind() == Z3_OP_MOD and \
                node.arg(0).arg(0).decl().kind() == Z3_OP_SUB and \
                node.arg(0).arg(1).decl().kind() == Z3_OP_ANUM and \
                node.arg(1).decl().kind() == Z3_OP_ANUM and \
                node.arg(1).as_long() == 0:
            # detected the ast root of a divisibility constraint

            subtr = node.arg(0).arg(0)
            assert subtr.decl().kind() == Z3_OP_SUB

            lhs = subtr.arg(0)
            rhs = subtr.arg(1)

            modulo = node.arg(0).arg(1)

            div_constraints.add((lhs, rhs, modulo))
            return

        assert node.decl().kind() != Z3_OP_MOD

        for child in node.children():

            child_wrapped = wrap_ast_ref(child)

            if child_wrapped in visited:
                continue

            visited.add(child_wrapped)

            detect_div_constraints(child)

    visited.add(wrap_ast_ref(phi))

    try:
        detect_div_constraints(phi)
    except (RecursionError, ctypes.ArgumentError):
        raise MonDecTestFailed("recursion error: stack overflow")

    return And([
        _same_div_transform_constraint(x_1, x_2, x, c)
        for c in div_constraints
    ])


def _run_z3_solver_with_timeout(s, timeout_ms: int):

    if timeout_ms > 0:
        s.set(timeout=timeout_ms)
        s.set(solver2_timeout=timeout_ms)

        smt_string = s.to_smt2()

        return run_z3_cli(smt_string, timeout_ms)

    return s.check()


def monadic_decomposable(f, x, b=None, timeout_ms=0) -> bool:

    if b is None:
        b = compute_bound(f)

    x_1, x_2 = Ints("x_1 x_2")

    sd = _same_div(x_1, x_2, x, f)

    try:
        vars_except_x = get_formula_variables(f)
    except TooDeepFormulaError:
        raise MonDecTestFailed("formula AST is too deep to traverse")

    vars_except_x.remove(wrap_ast_ref(x))

    mon_dec_formula = Exists([x_1, x_2, *[v.unwrap() for v in vars_except_x]], And(
        x_1 >= b,
        x_2 >= b,
        sd,
        substitute(f, (x, x_1)),
        Not(substitute(f, (x, x_2)))
    ))

    s = Solver()

    s.add(mon_dec_formula)

    result = _run_z3_solver_with_timeout(s, timeout_ms)

    if result == unknown:
        raise MonDecTestFailed("z3 solver returned unknown")

    return result == unsat


def _equiv(phi, x, a, b):
    """
    Constructs the formula describing the equivalence relation that is finite iff phi
    is monadically decomposable
    :param phi: the formula
    :param x: the variable we want to decompose on
    :param a: first argument for the equivalence
    :param b: second argument for the equivalence
    :return: formula describing the equivalence relation
    """

    try:
        phi_vars_except_x = get_formula_variables(phi)
    except TooDeepFormulaError:
        raise MonDecTestFailed("formula AST is too deep to traverse")

    phi_vars_except_x.remove(wrap_ast_ref(x))

    phi_vars_except_x_new = {}

    for v in phi_vars_except_x:
        phi_vars_except_x_new[v] = Int(v.unwrap().decl().name() + "_pr")

    phi_vars_except_x_new_subs = [(v.unwrap(), phi_vars_except_x_new[v]) for v in phi_vars_except_x]

    implication = Implies(
        And(
            substitute(phi, (x, a)),
            substitute(phi, (x, b), *phi_vars_except_x_new_subs)
        ),
        And(
            substitute(phi, (x, b)),
            substitute(phi, (x, a), *phi_vars_except_x_new_subs)
        )
    )

    return ForAll([*[v.unwrap() for v in phi_vars_except_x],
                   *[v for v in phi_vars_except_x_new.values()]], implication)


def monadic_decomposable_without_bound(f, x, bound_bound_hint=None, timeout_ms=0) -> bool:

    b, t, e = Ints("b t e")

    bound_hint_formula = [b <= bound_bound_hint] if bound_bound_hint is not None else []

    cf = Exists(
        [b],
        And(
            b >= 0,
            *bound_hint_formula,
            ForAll(
                [t],
                Implies(
                    t >= 0,
                    Exists(
                        [e],
                        And(
                            e >= 0,
                            e <= b,
                            _equiv(f, x, t, e)
                        )
                    )
                )
            )
        )
    )

    s = Solver()

    s.add(Not(cf))

    result = _run_z3_solver_with_timeout(s, timeout_ms)

    if result == unknown:
        raise MonDecTestFailed("z3 solver returned unknown")

    return result == unsat


if __name__ == "__main__":

    x, y, z = Ints("x y z")

    phi = And([
        x >= 0,
        y >= 0,
        z >= 0,
        x + 2 * y >= 5,
        z < 5,
        congruent(x, y, 2)
    ])

    b = compute_bound(phi)

    print("B =", b)

    if monadic_decomposable(phi, x, b):
        print("Monadically decomposable")
    else:
        print("Not monadically decomposable")

    print("=" * 30)
    print("Caution: this may fail with a timeout error")

    if monadic_decomposable_without_bound(phi, x, b, 5000):
        print("Monadically decomposable")
    else:
        print("Not monadically decomposable")
