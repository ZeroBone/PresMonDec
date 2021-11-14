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
