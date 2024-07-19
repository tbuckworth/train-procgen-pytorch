import functools as ft

import numpy as np
import torch


def _reduce(fn):
    def fn_(*args):
        return ft.reduce(fn, args)

    return fn_


binary_functions = {
    "*": _reduce(torch.mul),
    "+": _reduce(torch.add),
    "/": torch.div,
    "max": torch.max,
    "min": torch.min,
    "mod": torch.remainder,
    "heaviside": torch.heaviside,
}

binary_booleans = {
    "=": torch.eq,
    "!=": torch.ne,
    ">": torch.gt,
    "<": torch.lt,
    "<=": torch.le,
    ">=": torch.ge,
    r"/\\": torch.logical_and,
    r"\/": torch.logical_or,
    "!": torch.logical_not,
}

unary_functions = {
    "abs": torch.abs,
    "sign": torch.sign,
    # Note: May raise error for ints.
    "ceil": torch.ceil,
    "floor": torch.floor,
    "log": torch.log,
    "exp": torch.exp,
    "sqrt": torch.sqrt,
    "cos": torch.cos,
    "acos": torch.acos,
    "sin": torch.sin,
    "asin": torch.asin,
    "tan": torch.tan,
    "atan": torch.atan,
    "atan2": torch.atan2,
    # Note: May give NaN for complex results.
    "cosh": torch.cosh,
    "acosh": torch.acosh,
    "sinh": torch.sinh,
    "asinh": torch.asinh,
    "tanh": torch.tanh,
    "atanh": torch.atanh,
    "pow": torch.pow,
    # "real": torch.real,
    # "imag": torch.imag,
    # "angle": torch.angle,
    # Note: May raise error for ints and complexes
    # "erf": torch.erf,
    # "lgamma": torch.lgamma,
}
constants = {
    "half": (lambda: 0.5),
    "one": (lambda: 1.0),
}


#
# return {
#         # sympy.Mul: _reduce(torch.mul),
#         # sympy.Add: _reduce(torch.add),
#         sympy.Heaviside: torch.heaviside,
#         sympy.core.numbers.Half: (lambda: 0.5),
#         sympy.core.numbers.One: (lambda: 1.0),
#         sympy.logic.boolalg.Boolean: as_bool,
#         sympy.logic.boolalg.BooleanTrue: (lambda: True),
#         sympy.logic.boolalg.BooleanFalse: (lambda: False),
#         sympy.functions.elementary.piecewise.ExprCondPair: expr_cond_pair,
#         sympy.Piecewise: piecewise,
#         sympy.logic.boolalg.ITE: if_then_else,
#         sympy.core.numbers.Exp1: exp1,
#         sympy.exp: exp,
#         sympy.core.numbers.ComplexInfinity: inf,
#     }

class BaseNode:
    def __init__(self, x, name):
        self.x = x
        self.name = name

    def evaluate(self):
        return self.x

    def get_name(self):
        return self.name


class UnaryNode:
    def __init__(self, func, x1):
        self.func = func
        self.f = unary_functions.get(func)
        self.x1 = x1

    def evaluate(self, x1=None):
        if x1 is None:
            x1 = self.x1.evaluate()
        return self.f(x1)

    def get_name(self):
        return f"{self.func}({self.x1.get_name()})"


def check_style(func):
    if func in ["+", "/", "-", "*"]:
        return "mid"
    return "pre"


def get_output_type(func):
    #TODO: get this working
    return torch.FloatType



class BinaryNode:
    def __init__(self, func, x1, x2):
        self.func = func
        self.style = check_style(func)
        self.f = binary_functions.get(func)
        self.x1 = x1
        self.x2 = x2
        self.output_type = get_output_type(func)

    def evaluate(self, x1=None, x2=None):
        if x1 is None:
            x1 = self.x1.evaluate()
        if x2 is None:
            x2 = self.x2.evaluate()
        return self.f(x1, x2)

    def get_name(self):
        if self.style == "pre":
            return f"{self.func}({self.x1.get_name()},{self.x2.get_name()})"
        if self.style == "mid":
            return f"{self.x1.get_name()} {self.func} {self.x2.get_name()}"
        raise NotImplementedError("style must be one of (mid,pre)")

def combine_bin_funcs(base_vars, func_list, node_cons, max_funcs):
    #TODO: only pick inputs of the right type
    #   add in constants
    max_index = len(func_list) * len(base_vars) ** 2
    n_funcs = min(np.random.randint(max_index), max_funcs)
    vars = []
    for _ in range(n_funcs):
        key = np.random.choice(np.array(list(func_list.keys())))
        x1 = base_vars[np.random.randint(len(base_vars))]
        x2 = base_vars[np.random.randint(len(base_vars))]
        vars += [node_cons(key, x1, x2)]
    return vars

def combine_funcs(base_vars, func_list, node_cons):
    max_index = len(func_list) * len(base_vars)
    n_funcs = np.random.randint(max_index)
    indices = np.random.randint(max_index, size=n_funcs)
    ind = np.mod(indices, len(func_list))
    jnd = np.floor(indices / len(func_list))
    keys = np.array(list(func_list.keys()))[ind]
    return [node_cons(key, base_vars[int(jnd[i])]) for i, key in enumerate(keys)]


class FunctionTree:
    def __init__(self, x, y):
        self.max_active_vars = 100
        in_vars = torch.split(x, 1, 1)
        self.base_vars = [BaseNode(z, f"x{i}") for i, z in enumerate(in_vars)]
        self.all_vars = self.base_vars
        self.all_vars += combine_funcs(self.all_vars, unary_functions, UnaryNode)
        self.all_vars += combine_bin_funcs(self.all_vars, binary_functions, BinaryNode, max_funcs=100)

        return


def create_func(x, y):
    in_vars = torch.split(x, 1, 1)
    base_vars = [BaseNode(z, f"x{i}") for i, z in enumerate(in_vars)]
    x0 = BaseNode(in_vars[0], "x0")
    x2 = BaseNode(in_vars[2], "x2")
    u = UnaryNode("exp", x0)
    b = BinaryNode("*", u, x2)
    print(b.get_name())
    self = FunctionTree(x, y)
    return
