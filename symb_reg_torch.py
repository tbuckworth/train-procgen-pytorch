import functools as ft
from abc import ABC

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import linear_model
from torch import nn

from helper_local import softmax


def _reduce(fn):
    def fn_(*args):
        return ft.reduce(fn, args)

    return fn_


input_types = {
    "*": ["float", "int", "bool"],
    "+": ["float", "int", "bool"],
    "/": ["float", "int", "bool"],
    "max": ["float", "int", "bool"],
    "min": ["float", "int", "bool"],
    "mod": ["int", "bool"],
    "heaviside": ["float", "int", "bool"],
    "==": ["float", "int", "bool"],
    "!=": ["float", "int", "bool"],
    ">": ["float", "int", "bool"],
    "<": ["float", "int", "bool"],
    "<=": ["float", "int", "bool"],
    ">=": ["float", "int", "bool"],
    r"/\\": ["bool"],
    r"\/": ["bool"],
    "!": ["bool"],
    "abs": ["float", "int"],
    "sign": ["float", "int", "bool"],
    # Note: May raise error for ints.
    "ceil": ["float"],
    "floor": ["float"],
    "log": ["float", "int"],
    "exp": ["float", "int", "bool"],
    "sqrt": ["float", "int", "bool"],
    "cos": ["float", "int", "bool"],
    "acos": ["float", "int", "bool"],
    "sin": ["float", "int", "bool"],
    "asin": ["float", "int", "bool"],
    "tan": ["float", "int", "bool"],
    "atan": ["float", "int", "bool"],
    "atan2": ["float", "int", "bool"],
    # Note: May give NaN for complex results.
    "cosh": ["float", "int", "bool"],
    "acosh": ["float", "int", "bool"],
    "sinh": ["float", "int", "bool"],
    "asinh": ["float", "int", "bool"],
    "tanh": ["float", "int", "bool"],
    "atanh": ["float", "int", "bool"],
    "square": ["float", "int", "bool"],
    "cube": ["float", "int", "bool"],
}

output_types = {
    "*": "input",
    "+": "input",
    "/": "float",
    "max": "input",
    "min": "input",
    "mod": "int",
    "heaviside": "input",
    "==": "bool",
    "!=": "bool",
    ">": "bool",
    "<": "bool",
    "<=": "bool",
    ">=": "bool",
    r"/\\": "bool",
    r"\/": "bool",
    "!": "bool",
    "abs": "input",
    "sign": "input",
    # Note: May raise error for ints.
    "ceil": "int",
    "floor": "int",
    "log": "float",
    "exp": "float",
    "sqrt": "float",
    "cos": "float",
    "acos": "float",
    "sin": "float",
    "asin": "float",
    "tan": "float",
    "atan": "float",
    "atan2": "float",
    # Note: May give NaN for complex results.
    "cosh": "float",
    "acosh": "float",
    "sinh": "float",
    "asinh": "float",
    "tanh": "float",
    "atanh": "float",
    # "pow": "input",
    "square": "input",
    "cube": "input",
}

binary_functions = {
    "*": _reduce(torch.mul),
    "+": _reduce(torch.add),
    "/": torch.div,
    "max": torch.max,
    "min": torch.min,
    "mod": torch.remainder,
    # "heaviside": torch.heaviside,
    "atan2": torch.atan2,
    # "pow": torch.pow,
# }

# binary_booleans = {
    "==": torch.eq,
    "!=": torch.ne,
    ">": torch.gt,
    "<": torch.lt,
    "<=": torch.le,
    ">=": torch.ge,
    r"/\\": torch.logical_and,
    r"\/": torch.logical_or,
}

unary_functions = {
    "!": torch.logical_not,
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
    # Note: May give NaN for complex results.
    "cosh": torch.cosh,
    "acosh": torch.acosh,
    "sinh": torch.sinh,
    "asinh": torch.asinh,
    "tanh": torch.tanh,
    "atanh": torch.atanh,
    "square": torch.square,
    "cube": lambda x: torch.pow(x, 3),
    "relu": torch.relu,
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

class Node(ABC):
    loss = None

    def __init__(self):
        if self.output_type == "input":
            self.output_type = self.input_type
        self.value = self.evaluate()
        self.min_loss = np.inf

    def compute_loss(self, loss_fn, y):
        with torch.no_grad():
            self.loss = loss_fn(y, self.value).cpu().numpy()
        self.min_loss = np.min([self.loss, self.min_loss])
        if np.isnan(self.loss) or np.isinf(self.loss):
            return self.loss
        for n in self.super_nodes:
            n.min_loss = np.min([self.loss, n.min_loss])
        return self.loss

    def evaluate(self):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError


class ScalarNode(Node):
    def __init__(self, like, name, value):
        self.x = torch.full_like(like, value).to(device=like.device)
        self.name = name
        self.input_type = None
        self.output_type = "float"
        self.super_nodes = []
        self.complexity = 0
        self.birth = np.inf
        super().__init__()

    def forward(self, x):
        return self.x.squeeze()

    def evaluate(self):
        return self.x

    def get_name(self):
        return self.name


class BaseNode(Node):
    def __init__(self, x, name, ind):
        self.ind = ind
        self.x = x
        self.name = name
        self.input_type = None
        self.output_type = "float"
        self.super_nodes = []
        self.complexity = 0
        self.birth = np.inf
        super().__init__()

    def forward(self, x):
        return x[..., self.ind]

    def evaluate(self):
        return self.x

    def get_name(self):
        return self.name


class UnaryNode(Node):
    def __init__(self, func, date, x1):
        self.birth = date
        self.func = func
        self.f = unary_functions.get(func)
        self.x1 = x1
        self.complexity = x1.complexity + 1
        self.super_nodes = [self.x1]
        self.input_type = x1.output_type
        self.output_type = output_types[func]
        super().__init__()

    def forward(self, x):
        return self.f(self.x1.forward(x))

    def evaluate(self, x1=None):
        if x1 is None:
            x1 = self.x1.evaluate()
        try:
            return self.f(x1)
        except RuntimeError as e:
            raise e

    def get_name(self):
        return f"{self.func}({self.x1.get_name()})"


def check_style(func):
    if func in ["+", "/", "-",
                "*", "==", "!=",
                ">", "<", "<=",
                ">=", r"/\\", r"\/"]:
        return "mid"
    return "pre"


def get_output_type(xs):
    all_types = [x.output_type for x in xs]
    if "float" in all_types:
        return "float"
    if "int" in all_types:
        return "int"
    if "bool" in all_types:
        return "bool"


class BinaryNode(Node):
    def __init__(self, func, date, x1, x2):
        self.birth = date
        self.func = func
        self.style = check_style(func)
        self.f = binary_functions.get(func)
        self.x1 = x1
        self.x2 = x2
        self.complexity = x1.complexity + x2.complexity + 2
        self.super_nodes = [self.x1, self.x2]
        self.input_type = get_output_type([x1, x2])
        self.output_type = output_types[func]
        super().__init__()

    def forward(self, x):
        return self.f(self.x1.forward(x), self.x2.forward(x))

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


class ConditionalNode(Node):
    def __init__(self, date, cond, x1, x2):
        self.birth = date
        self.cond = cond
        self.x1 = x1
        self.x2 = x2
        self.super_nodes = [cond, x1, x2]
        self.complexity = x1.complexity + x2.complexity + cond.complexity + 1
        self.input_type = get_output_type([x1, x2])
        self.output_type = self.input_type
        super().__init__()

    def forward(self, x):
        return torch.where(self.cond.forward(x), self.x1.forward(x), self.x2.forward(x))

    def evaluate(self, cond=None, x1=None, x2=None):
        if cond is None:
            cond = self.cond.evaluate()
        if x1 is None:
            x1 = self.x1.evaluate()
        if x2 is None:
            x2 = self.x2.evaluate()
        cond = cond.to(bool)
        return torch.where(cond, x1, x2)

    def get_name(self):
        return f"({self.cond.get_name()}) ? {self.x1.get_name()} : {self.x2.get_name()}"


class SymbolicFunction(nn.Module):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.name = node.get_name()

    def forward(self, x):
        return self.node.forward(x)


def combine_funcs(base_vars, loss_fn, y, date, max_funcs, n_inputs=1, max_complexity=20):
    if n_inputs == 1:
        node_cons = UnaryNode
        func_list = unary_functions
    elif n_inputs == 2:
        node_cons = BinaryNode
        func_list = binary_functions
    else:
        raise NotImplementedError("n_inputs must be 1 or 2")
    max_index = len(func_list) * len(base_vars) ** n_inputs
    n_funcs = min(np.random.randint(max_index), max_funcs)
    vars = []
    for _ in range(n_funcs):
        key = np.random.choice(np.array(list(func_list.keys())))
        temp_vars = [v for v in base_vars if v.output_type in input_types[key] and v.complexity < max_complexity]
        if len(temp_vars) == 0:
            continue
        complexity = np.array([v.complexity for v in temp_vars])
        r = complexity.max() - complexity + 1
        p = r / r.sum()
        xs = np.random.choice(temp_vars, n_inputs, p=p).tolist()
        # xs = [temp_vars[np.random.randint(len(temp_vars))] for _ in range(n_inputs)]
        vars += [node_cons(key, date, *xs)]
        loss = vars[-1].compute_loss(loss_fn, y)
        if np.isnan(loss) or np.isinf(loss):
            vars.pop()
    return vars


# def combine_un_funcs(base_vars, func_list, node_cons, max_funcs):
#     max_index = len(func_list) * len(base_vars)
#     n_funcs = min(np.random.randint(max_index), max_funcs)
#     vars = []
#     for _ in range(n_funcs):
#         key = np.random.choice(np.array(list(func_list.keys())))
#         temp_vars = [v for v in base_vars if v.output_type in input_types[key]]
#         x1 = temp_vars[np.random.randint(len(temp_vars))]
#         vars += [node_cons(key, x1)]
#     return vars
#
#
#
# def combine_funcs(base_vars, func_list, node_cons):
#     max_index = len(func_list) * len(base_vars)
#     n_funcs = np.random.randint(max_index)
#     indices = np.random.randint(max_index, size=n_funcs)
#     ind = np.mod(indices, len(func_list))
#     jnd = np.floor(indices / len(func_list))
#     keys = np.array(list(func_list.keys()))[ind]
#     return [node_cons(key, base_vars[int(jnd[i])]) for i, key in enumerate(keys)]


class FunctionTree:
    def __init__(self, x, y, loss_fn,
                 binary_funcs=None,
                 unary_funcs=None,
                 ):
        self.binary_funcs = ["/", "max", "min", "mod", "atan2"] if binary_funcs is None else binary_funcs
        self.unary_funcs = unary_functions.keys() if unary_funcs is None else unary_funcs
        self.stls_vars = []
        self.x = x
        self.y = y
        self.loss_fn = loss_fn
        self.rounds = 2
        self.max_active_vars = 100
        in_vars = torch.split(x, 1, 1)
        self.n_base = len(in_vars) + 1
        self.base_vars = [BaseNode(z, f"x{i}", i) for i, z in enumerate(in_vars)] + [ScalarNode(in_vars[0], "1", 1.)]
        _ = [b.compute_loss(loss_fn, y) for b in self.base_vars]
        self.all_vars = self.base_vars
        self.loss = np.array([])
        self.date = 0
        self.compute_stls()

    def evolve(self, pop_size):
        for i in range(self.rounds):
            self.all_vars += self.combine_funcs(max_funcs=500, n_inputs=1)
            self.all_vars += self.combine_funcs(max_funcs=200, n_inputs=2)
            self.all_vars += self.add_conditionals(max_funcs=100)
        self.compute_stls()
        self.date += 1
        min_losses = np.array([x.min_loss for x in self.all_vars])
        complexity = np.array([x.complexity for x in self.all_vars])
        ind = self.filter_population(min_losses, complexity, pop_size)
        self.all_vars = np.array(self.all_vars)[ind].tolist()
        self.loss = np.append(self.loss, np.min(min_losses))

    def train(self, pop_size, epochs):
        for epoch in range(epochs):
            self.evolve(pop_size)
            print(f"Loss at epoch {epoch}: {self.loss[-1]}")

    def get_best(self):
        complete_vars = self.all_vars + self.stls_vars
        losses = np.array([x.loss for x in complete_vars if x.loss is not None])
        best_node = complete_vars[np.argmin(losses)]
        print(best_node.get_name())
        model = SymbolicFunction(best_node)
        return model

    def add_conditionals(self, max_funcs, max_complexity=20):
        n_funcs = np.random.randint(max_funcs)
        conds = [v for v in self.all_vars if v.output_type == "bool" and v.complexity < max_complexity]
        new_vars = []
        selected_conds = self.sample_by_inverse_complexity(n_funcs, conds)
        for cond in selected_conds:
            poss_x1 = [v for v in self.all_vars if v.complexity < max_complexity]
            x1 = self.sample_by_inverse_complexity(1, poss_x1)[0]

            poss_x2 = [v for v in self.all_vars if
                       v.output_type == x1.output_type and v.complexity < max_complexity]
            x2 = self.sample_by_inverse_complexity(1, poss_x2)[0]
            new_vars += [ConditionalNode(self.date, cond, x1, x2)]
            loss = new_vars[-1].compute_loss(self.loss_fn, self.y)
            if np.isnan(loss) or np.isinf(loss):
                new_vars.pop()
        return new_vars

    def combine_funcs(self, max_funcs, n_inputs=1, max_complexity=20):
        if n_inputs == 1:
            node_cons = UnaryNode
            func_list = {k: v for k, v in unary_functions.items() if k in self.unary_funcs}
        elif n_inputs == 2:
            node_cons = BinaryNode
            func_list = {k: v for k, v in binary_functions.items() if k in self.binary_funcs}
        else:
            raise NotImplementedError("n_inputs must be 1 or 2")
        max_index = len(func_list) * len(self.all_vars) ** n_inputs
        n_funcs = min(np.random.randint(max_index), max_funcs)
        new_vars = []
        for _ in range(n_funcs):
            key = np.random.choice(np.array(list(func_list.keys())))
            temp_vars = [v for v in self.all_vars if
                         v.output_type in input_types[key] and v.complexity < max_complexity]
            if len(temp_vars) == 0:
                continue
            xs = self.sample_by_inverse_complexity(n_inputs, temp_vars)
            new_vars += [node_cons(key, self.date, *xs)]
            loss = new_vars[-1].compute_loss(self.loss_fn, self.y)
            if np.isnan(loss) or np.isinf(loss):
                new_vars.pop()
        return new_vars

    def sample_by_inverse_complexity(self, n_inputs, temp_vars):
        complexity = np.array([v.complexity for v in temp_vars])
        r = complexity.max() - complexity + 1
        p = r / r.sum()
        xs = np.random.choice(temp_vars, n_inputs, p=p).tolist()
        return xs

    def print_everything(self):
        _ = [print(v.get_name()) for v in self.all_vars]

    def filter_population(self, min_losses, complexity, pop_size):
        if pop_size >= len(min_losses) - self.n_base:
            return np.arange(len(min_losses))
        min_losses = min_losses[self.n_base:]
        # Rank descending from 1:
        r = (-min_losses).argsort().argsort() + 1
        r -= complexity[self.n_base:]
        # Probability of survival is proportional to rank
        # and scaled to target pop_size
        p = r / np.sum(r) * (pop_size - 20)
        # Best 20 are kept with 100% prob
        p[r > np.max(r) - 20] = 1.0
        base_ind = np.full((self.n_base,), True)
        rem_ind = np.random.random((p.shape)) < p
        oldest = np.argmin([x.birth for x in self.all_vars])
        full_index = np.concatenate((base_ind, rem_ind))
        full_index[oldest] = False
        return full_index

    def STLS(self, threshold=0.01, thresh_inc=0.05, max_thresh=100, n_param_target=5):
        dictionary = [v.evaluate() for v in self.all_vars]
        d = torch.cat(dictionary, dim=1)
        dtmp = d
        idx = np.arange(d.shape[-1])
        clf = linear_model.LinearRegression(fit_intercept=False)
        min_loss = np.inf
        last_coef = None
        while dtmp.shape[-1] > n_param_target and threshold < max_thresh:
            n_cmp = -1
            threshold += thresh_inc
            while n_cmp != dtmp.shape[-1]:
                clf.fit(dtmp, self.y)
                coef = clf.coef_
                y_hat = clf.predict(dtmp)
                loss = self.loss_fn(torch.Tensor(y_hat).to(device=self.y.device), self.y)
                # print(f"Loss: {loss:.4f}")
                if loss < min_loss:
                    best_idx = idx
                    best_coef = coef
                    min_loss = loss
                flt = np.abs(coef) > threshold
                if not np.any(flt):
                    if last_coef is None:
                        return self.STLS(threshold=threshold / 10, thresh_inc=thresh_inc / 10)
                    return last_coef, idx, best_coef, best_idx
                idx = idx[flt]
                n_cmp = dtmp.shape[-1]
                dtmp = dtmp[..., flt]
                last_coef = coef
            # if np.min(np.abs(coef)) > 1000:
            #     break
        return coef, idx, best_coef, best_idx

    def compute_stls(self, keep_overfit=False):
        coef, idx, best_coef, best_idx = self.STLS()
        final_var, new_vars = self.convert_to_func(coef, idx)
        print(final_var.get_name())
        final_var.compute_loss(self.loss_fn, self.y)
        # self.all_vars += new_vars + [final_var]
        self.stls_vars += [final_var]
        if keep_overfit:
            final_var, new_vars = self.convert_to_func(best_coef, best_idx)
            print(final_var.get_name())
            final_var.compute_loss(self.loss_fn, self.y)
            self.all_vars += new_vars + [final_var]

    def convert_to_func(self, coef, idx):
        scalars = [ScalarNode(self.base_vars[0].x, f"{c:.2f}", c) for c in coef]
        v = np.array(self.all_vars)[idx].tolist()
        new_vars = [BinaryNode("*", self.date, c, f) for c, f in zip(scalars, v)]
        final_var = new_vars[0]
        for v in new_vars[1:]:
            final_var = BinaryNode("+", self.date, final_var, v)
        return final_var, new_vars


def create_func(x, y):
    in_vars = torch.split(x, 1, 1)
    base_vars = [BaseNode(z, f"x{i}", i) for i, z in enumerate(in_vars)]
    x0 = BaseNode(in_vars[0], "x0")
    x2 = BaseNode(in_vars[2], "x2")
    u = UnaryNode("exp", x0)
    b = BinaryNode("*", u, x2)
    print(b.get_name())


def run_tree(x, y, pop_size, epochs):
    tree = FunctionTree(x, y, torch.nn.MSELoss())
    tree.train(pop_size=pop_size, epochs=epochs)
    return tree
