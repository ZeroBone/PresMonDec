from z3 import *


def compute_bound(f) -> int:

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

        if node.decl().name() == "=":
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


def same_div(x_1, x_2):
    # TODO
    pass


def monadic_decomposable(f, x) -> bool:

    b = compute_bound(f)

    print("B =", b)

    # TODO

    print(f.num_args())
    print(f.arg(0))
    print(f.children())
    print(f.arg(0).children())
    print(f.arg(0).arg(0))

    return True


if __name__ == "__main__":

    x, y, z = Ints("x y z")

    phi = And([
        x >= 0,
        y >= 0,
        z >= 0,
        x + 2 * y >= 5,
        z < 5,
        (x - y) % 2 == 0
    ])

    if monadic_decomposable(phi, x):
        print("Monadically decomposable")
    else:
        print("Not monadically decomposable")
