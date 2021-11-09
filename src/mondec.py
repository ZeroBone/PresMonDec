"""
Author: Margus Veanes, with changes by Anthony Widjaja Lin
"""

from z3 import *


def nu_ab(R, x, y, a, b):
    x_ = [Const("x_%d" % i, x[i].sort()) for i in range(len(x))]
    y_ = [Const("y_%d" % i, y[i].sort()) for i in range(len(y))]
    return Or(Exists(y_, R(x + y_) != R(a + y_)), Exists(x_, R(x_ + y) != R(x_ + b)))


def is_unsat(fml):
    s = Solver()
    s.add(fml)
    return unsat == s.check()


def last_sat(s, m, fmls):
    if len(fmls) == 0:
        return m
    s.push()
    s.add(fmls[0])
    if s.check() == sat:
        m = last_sat(s, s.model(), fmls[1:])
    s.pop()
    return m


def mondec(R, variables):
    print(variables)
    phi = R(variables)
    if len(variables) == 1:
        return phi
    l = int(len(variables) / 2)
    x, y = variables[0:l], variables[l:]

    def dec(nu, pi):
        if is_unsat(And(pi, phi)):
            return BoolVal(False)
        if is_unsat(And(pi, Not(phi))):
            return BoolVal(True)
        fmls = [BoolVal(True), phi, pi]
        # try to extend nu
        m = last_sat(nu, None, fmls)
        # nu must be consistent
        assert (m is not None)
        a = [m.evaluate(z, True) for z in x]
        b = [m.evaluate(z, True) for z in y]
        psi_ab = And(R(a + y), R(x + b))
        phi_a = mondec(lambda z: R(a + z), y)
        phi_b = mondec(lambda z: R(z + b), x)
        nu.push()
        # exclude: x~a and y~b
        nu.add(nu_ab(R, x, y, a, b))
        t = dec(nu, And(pi, psi_ab))
        f = dec(nu, And(pi, Not(psi_ab)))
        nu.pop()
        return If(And(phi_a, phi_b), t, f)

    # nu is initially true
    return dec(Solver(), BoolVal(True))


def test_mondec(k):
    R = lambda v: And(v[1] > 0, (v[1] & (v[1] - 1)) == 0,
                      (v[0] & (v[1] % ((1 << k) - 1))) != 0)
    bvs = BitVecSort(2 * k)  # use 2k-bit bitvectors
    x, y = Consts('x y', bvs)
    res = mondec(R, [x, y])
    assert (is_unsat(res != R([x, y])))  # check correctness
    print("mondec1(", R([x, y]), ") =", res)


def test_mondec1(k):
    R = lambda v: And(v[0] + v[1] >= k, v[0] >= 0, v[1] >= 0)
    x, y = Consts('x y', IntSort())
    res = mondec(R, [x, y])
    assert (is_unsat(res != R([x, y])))  # check correctness
    print("mondec1(", R([x, y]), ") =", res)


def test_mondec2(k):
    R = lambda v: Or(Or([And(v[0] <= i + 2, v[0] >= i, v[1] >= i, v[1] <= i + 2) for
                         i in range(1, k)]),
                     And(v[0] + v[1] == k, v[0] >= 0, v[1] >= 0))
    x, y = Consts('x y', IntSort())
    res = mondec(R, [x, y])
    assert (is_unsat(res != R([x, y])))  # check correctness
    print("mondec1(", R([x, y]), ") =", res)


def test_mondec3(k):
    R = lambda v: Or([And(v[0] <= i + 2, v[0] >= i, v[1] >= i, v[1] <= i + 2) for
                      i in range(1, k)])
    x, y = Consts('x y', IntSort())
    res = mondec(R, [x, y])
    assert (is_unsat(res != R([x, y])))  # check correctness
    print("mondec1(", R([x, y]), ") =", res)


test_mondec3(500)
