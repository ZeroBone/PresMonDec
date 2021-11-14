from z3 import *


def _compute_bound(f) -> int:

    visited = set()

    def ast_visitor(node):

        if is_int_value(node):
            return node.as_long().bit_length(), 0, 0

        if is_const(node) and node.decl().kind() == Z3_OP_UNINTERPRETED:
            # this is a variable
            return 0, 0, 1

        if not is_app(node):
            raise Exception("The AST of the formula contains a leaf that is not a integer constant")

        max_d = 1
        linear_equality_count = 0
        variable_count = 0

        for child in node.children():

            if child in visited:
                continue

            visited.add(child)

            d, n, m = ast_visitor(child)

            max_d = max(max_d, d)
            linear_equality_count += n
            variable_count += m

        if node.decl().kind() == Z3_OP_EQ:
            # found an equality
            linear_equality_count += 1

        return max_d, linear_equality_count, variable_count

    if f.decl().name() == "or":

        d = 1
        n = 0
        m = 0

        for child in f.children():

            if child in visited:
                continue

            visited.add(child)

            cur_d, cur_n, cur_m = ast_visitor(child)

            d = max(d, cur_d)
            n = max(n, cur_n)
            m += cur_m

    else:
        d, n, m = ast_visitor(f)

    return 8 << (d * n * m)


def congruent(lhs, rhs, modulo):
    return (lhs - rhs) % modulo == 0


# def _substitute_var(f, x, y):
#
#     if is_const(f) and f.decl().kind() == Z3_OP_UNINTERPRETED and eq(f, x):
#         print("Detected variable x", f)
#         return y
#
#     if len(f.children()) == 0:
#         return f
#
#     return f.decl()([_substitute_var(child, x, y) for child in f.children()])


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

        for child in node.children():

            if child in visited:
                continue

            visited.add(child)

            detect_div_constraints(child)

    visited.add(phi)
    detect_div_constraints(phi)

    print("Divisibility constraints found:", div_constraints)

    return And([
        _same_div_transform_constraint(x_1, x_2, x, c)
        for c in div_constraints
    ])


def monadic_decomposable(f, x) -> bool:

    b = _compute_bound(f)

    print("B =", b)

    x_1, x_2 = Ints("x_1 x_2")

    s = _same_div(x_1, x_2, x, f)

    print(s)

    # TODO

    return True


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

    print(phi)

    if monadic_decomposable(phi, x):
        print("Monadically decomposable")
    else:
        print("Not monadically decomposable")
