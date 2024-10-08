import functools as ft
from abc import ABC

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pysindy import STLSQ
from scipy.signal import argrelextrema
# from sklearn import linear_model
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KernelDensity
from torch import nn

from double_graph_sr import generate_data
from helper_local import softmax
from symbolic_regression import load_nn_policy


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
    "relu": ["float", "int"],
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
    "relu": "input",
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
    super_nodes = []
    output_type = None
    input_type = None

    def __init__(self):
        self.flt = None
        self.n_outliers = None
        if self.output_type == "input":
            self.output_type = self.input_type
        self.value = self.evaluate()
        self.min_loss = np.inf
        self.std = torch.std(self.value.to(float)).item()

    def compute_loss(self, loss_fn, y):
        with torch.no_grad():
            y_hat = self.value.squeeze()
            y = y.squeeze()
            self.corr = torch.corrcoef(torch.cat((y_hat, y))).abs().item()
            if self.output_type != "bool":
                e = y_hat - y
                self.n_outliers = (torch.abs(e) - 2 * e.std() > 0).sum().item()
            self.loss = loss_fn(y, y_hat).item()
        self.min_loss = np.min([self.loss, self.min_loss])
        if np.isnan(self.loss) or np.isinf(self.loss):
            return self.loss
        if self.output_type != "bool":
            with torch.no_grad():
                pairwise_loss = e ** 2
                split_points = get_kde_minima_1d(pairwise_loss)
                if len(split_points) == 1:
                    self.flt = pairwise_loss < split_points.item()
        for n in self.super_nodes:
            n.min_loss = np.min([self.loss, n.min_loss])
        return self.loss

    def evaluate(self):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError


class ScalarNode(Node):
    def __init__(self, like, name, value):
        self.scalar = value
        self.x = torch.full_like(like, value).to(device=like.device)
        self.name = name
        self.input_type = None
        self.output_type = "float"
        self.super_nodes = []
        self.complexity = 0
        self.birth = np.inf
        super().__init__()

    def forward(self, x):
        return torch.full((x.shape[0], *self.x.shape[1:]), self.scalar).squeeze().to(device=x.device)

    def evaluate(self):
        return self.x

    def get_name(self):
        return self.name


class BaseNode(Node):
    def __init__(self, x, name, ind, corr):
        self.ind = ind
        self.x = x
        self.name = name
        self.input_type = None
        self.output_type = "float"
        self.super_nodes = []
        self.complexity = 0  # 1 - corr.item()
        self.birth = np.inf
        self.corr = corr.item()
        super().__init__()

    def forward(self, x):
        return x[..., self.ind]

    def evaluate(self):
        return self.x

    def get_name(self):
        return self.name


def unary_node_cons(func, date, x1):
    f = unary_functions.get(func)
    if type(x1) is ScalarNode:
        val = f(x1.value.unique()).item()
        return ScalarNode(like=x1.value, name=f"{val:.2f}", value=val)
    return UnaryNode(func, date, x1)


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


def get_kde_minima_1d(a):
    a = a.cpu()
    # y_hat = np.random.random(y.shape)
    # y_hat = -y
    # y_hat[::2] = y[::2]
    # a = (y_hat-y.cpu().numpy())**2
    a = a.reshape(-1, 1)
    # a = np.array([10, 11, 9, 23, 21, 11, 45, 20, 11, 12]).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian').fit(a)
    s = np.linspace(a.min(), a.max(), 50)
    # s = a
    e = kde.score_samples(s.reshape(-1, 1))
    # plt.plot(s, e)
    # plt.show()
    # mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    mi = argrelextrema(e, np.less)[0]
    # print(mi)
    return s[mi]


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
        self.complexity = x1.complexity + x2.complexity + cond.complexity + 2
        self.input_type = get_output_type([x1, x2])
        self.output_type = self.input_type
        super().__init__()

    def forward(self, x):
        try:
            return torch.where(self.cond.forward(x).to(bool), self.x1.forward(x), self.x2.forward(x))
        except RuntimeError as e:
            raise e

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
        node_cons = unary_node_cons
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
                 max_complexity=20,
                 validation_ratio=0.1,
                 use_sindy=True,
                 x_indices=None):
        y = y.squeeze()
        self.x_indices = np.array(x_indices)
        self.use_sindy = use_sindy
        self.max_complexity = max_complexity
        self.binary_funcs = ["*", "/", "max", "min", "mod", "atan2"] if binary_funcs is None else binary_funcs
        self.unary_funcs = unary_functions.keys() if unary_funcs is None else unary_funcs
        self.stls_vars = []

        order = y.argsort()
        n = torch.ceil(order.max() * validation_ratio / 2)
        flt = torch.bitwise_or(order <= n, order >= order.max() - n)

        self.train_x = x[~flt]
        self.train_y = y[~flt]
        self.val_x = x[flt]
        self.val_y = y[flt]
        self.y_std = torch.std(y).item()
        self.corrs = torch.corrcoef(torch.cat((x.T, y.unsqueeze(-1).T)))[-1].abs()
        self.loss_fn = loss_fn
        self.rounds = 1
        self.max_active_vars = 100
        in_vars = torch.split(self.train_x, 1, 1)
        self.n_base = len(in_vars) + 1
        self.base_vars = [BaseNode(z, f"x{i}", i, self.corrs[i]) for i, z in enumerate(in_vars)] + [
            ScalarNode(in_vars[0], "1", 1.)]
        _ = [b.compute_loss(loss_fn, self.train_y) for b in self.base_vars]
        self.all_vars = self.base_vars
        self.loss = np.array([])
        self.date = 0
        if self.x_indices is not None:
            self.all_vars += self.index_vars()
        self.all_vars += self.all_combos()
        self.remove_duplicates()
        self.compute_stls()
        self.all_vars += self.all_combos()
        self.all_vars = self.filter_vars()
        self.compute_stls()

    def evolve(self, pop_size, find_splitpoints=True):
        # TODO: add mutation?
        for i in range(self.rounds):
            self.all_vars += self.combine_funcs(max_funcs=100, n_inputs=1)
            self.remove_duplicates()
            self.all_vars += self.combine_funcs(max_funcs=500, n_inputs=2)
            self.remove_duplicates()
            self.all_vars += self.add_conditionals(max_funcs=100)
        self.all_vars = self.filter_vars()
        self.compute_stls()
        if find_splitpoints:
            # self.alternative_splitpoint()
            self.all_vars += self.alternative_splitpoint()
            self.remove_duplicates()
        self.date += 1
        min_losses = np.array([x.min_loss for x in self.all_vars])
        # complexity = np.array([x.complexity for x in self.all_vars])
        # ind = self.filter_population(min_losses, complexity, pop_size)
        # self.all_vars = np.array(self.all_vars)[ind].tolist()
        self.loss = np.append(self.loss, np.min(min_losses))

    def train(self, pop_size, epochs, find_split_points=True):
        for epoch in range(epochs):
            self.evolve(pop_size, find_split_points)
            print(f"Loss at epoch {epoch}: {self.loss[-1]}")
        self.compute_val_loss()

    def get_best(self):
        complete_vars = [x for x in self.all_vars + self.stls_vars if self.condition(x)]
        losses = np.array([x.val_loss for x in complete_vars])
        best_node = complete_vars[np.argmin(losses)]
        print(best_node.get_name())
        model = SymbolicFunction(best_node)
        return model

    def condition(self, x):
        if self.y_std > 0.:
            return x.std > 0. and x.loss is not None
        return x.loss is not None

    def analyze(self, x, y):
        all_vars = [x for x in self.all_vars + self.stls_vars if x.n_outliers is not None and self.condition(x)]
        d = [[x.loss, x.n_outliers, x.std, x.complexity, x.val_loss] for x in all_vars]
        df = pd.DataFrame(d, columns=["loss", "n_outliers", "std", "complexity", "val_loss"])
        df[df["std"] > 0].plot.scatter("loss", "complexity", logx=True, logy=True)
        plt.show()

        df.loc[df.val_loss.rank() <= 10]

        ranks = df.drop(columns="std").rank()

        rank = df.drop(columns="std").rank().mean(axis=1)

        y_hat = all_vars[rank.argmin()].forward(x)
        y_hat = all_vars[df.loss.argmin()].forward(x)
        y_hat_val = all_vars[df.val_loss.argmin()].forward(x)

        plt.scatter(y, y_hat_val)

        y_hat = all_vars[203].forward(x)
        plt.scatter(y, y_hat)
        plt.scatter(y, y)
        plt.show()

        df.loc[rank.argmin()]

        df.loc[df.complexity.argmin()]
        df.loc[df.loss.argmin()]
        df.loc[df.val_loss.argmin()]
        df.loc[df.n_outliers.argmin()]

        min_c = df[df["std"] > 0].complexity.min()
        df.loc[df.complexity == min_c]

        all_vars[4].get_name()

    def add_conditionals(self, max_funcs, max_complexity=20):
        # n_funcs = np.random.randint(max_funcs)
        n_funcs = max_funcs
        conds = [v for v in self.all_vars if v.output_type == "bool" and v.complexity < max_complexity]
        new_vars = []
        if conds == []:
            return new_vars
        selected_conds = self.sample_by_inverse_complexity(n_funcs, conds)
        for cond in selected_conds:
            poss_x1 = [v for v in self.all_vars if v.complexity < max_complexity]
            x1 = self.sample_by_inverse_complexity(1, poss_x1)[0]

            poss_x2 = [v for v in self.all_vars if
                       v.output_type == x1.output_type and v.complexity < max_complexity]
            x2 = self.sample_by_inverse_complexity(1, poss_x2)[0]
            new_vars += [ConditionalNode(self.date, cond, x1, x2)]
            self.filter_loss(new_vars)
        return new_vars

    def alternative_splitpoint(self, score_threshold=0.8):
        tmp_vars = self.stls_vars + self.all_vars
        # works from existing bool_vars, but new bool_vars could be searched for or added
        bool_vars = [v for v in self.all_vars if v.output_type == "bool"]
        if len(bool_vars) == 0:
            return []
        all_bools = torch.cat([v.value for v in bool_vars], axis=-1)
        losses = self.piecewise_loss(tmp_vars)
        threshold = 0
        low_loss_bool = losses <= threshold

        n = len(low_loss_bool)
        order = np.argsort(n - low_loss_bool.sum(0))
        tmp_vars = np.array(tmp_vars)[order].tolist()
        low_loss_bool = low_loss_bool[:, order]
        nodes = []
        for i, v in enumerate(tmp_vars):
            f1 = low_loss_bool[:, i]
            if i > 10 or f1.sum() / n < 0.1:
                break
            f2 = low_loss_bool[:, (i + 1):]
            totals = (f1 + f2.T).sum(1) / n
            matches = totals > score_threshold
            if sum(matches) == 0:
                continue
            full_match_fltr = torch.cat([torch.BoolTensor([False for _ in range(i + 1)]), matches])
            matched_low_loss_bools = low_loss_bool[:, full_match_fltr]

            all_bools_expanded = all_bools.unsqueeze(-1).tile(1, 1, matched_low_loss_bools.shape[1])
            bool_scores = (all_bools_expanded == matched_low_loss_bools.unsqueeze(1)).sum(0) / n
            try:
                score = bool_scores.max()
            except RuntimeError as e:
                raise e
            if score > score_threshold:
                ind = bool_scores.argmax().item()
                cond_idx = ind // bool_scores.shape[1]
                alt_idx = ind % bool_scores.shape[1]
                try:
                    alt_v = np.array(tmp_vars)[full_match_fltr][alt_idx]
                except Exception as e:
                    raise e
                cond = bool_vars[cond_idx]
                nodes += [ConditionalNode(self.date, cond, v, alt_v)]
                self.filter_loss(nodes)
        return nodes

    def find_splitpoint_conditionals(self, score_threshold=0.8):
        # TODO: consider split_vars with low loss in one split and low complexity more
        split_vars = [v for v in self.stls_vars + self.all_vars if
                      v.flt is not None and v.complexity < self.max_complexity]
        # split_vars = np.array(split_vars)[np.argsort([v.complexity for v in split_vars])].tolist()
        if len(split_vars) == 0:
            return []
        all_loss = self.piecewise_loss(self.stls_vars + self.all_vars)
        func_list = {k: v for k, v in binary_functions.items() if k in self.binary_funcs and output_types[k] == "bool"}
        nodes = []
        for v in split_vars:
            cond, score = self.split_node(v, func_list)
            if score > score_threshold:
                nodes += [cond]
                cond.compute_loss(self.loss_fn, self.train_y)
                losses = all_loss[~v.flt].sum(0)
                nodes += [ConditionalNode(self.date, cond, v, (self.stls_vars + self.all_vars)[losses.argmin()])]
                self.filter_loss(nodes)
        return nodes

    def piecewise_loss(self, var_list):
        all_vals = torch.cat([v.value for v in var_list], axis=-1)
        all_loss = ((all_vals.T - self.train_y) ** 2).T
        return all_loss

    def split_node(self, v, func_list):
        # True is low loss, False is high loss
        split_point = None
        g1 = v.value[v.flt]
        g2 = v.value[~v.flt]
        # gt = g1 > g2
        if g2.max() < g1.min():
            split_point = (g2.max() + g1.min()).item() / 2
            func = ">="
        if g1.max() < g2.min():
            split_point = (g1.max() + g2.min()).item() / 2
            func = "<="
        if split_point is not None:
            return BinaryNode(func, self.date, v, ScalarNode(v.value, f"{split_point:.2f}", split_point)), 1.

        nodes = []
        tmp_vars = [x for x in self.stls_vars + self.all_vars if x.output_type != "bool" and x != v]
        for f in func_list:
            for t in tmp_vars:
                nodes += [BinaryNode(f, self.date, v, t)]
                nodes += [unary_node_cons("!", self.date, nodes[-1])]
                #
                # for t0 in tmp_vars:
                #     nodes += [BinaryNode(f, self.date, t0, t)]

        vals = torch.cat([v.value for v in nodes], axis=-1)
        matches = (v.flt == vals.T).sum(axis=-1)
        return nodes[matches.argmax()], matches.max().item() / len(v.flt)

    def all_combos(self):
        node_cons = unary_node_cons
        func_list = {k: v for k, v in unary_functions.items() if k in self.unary_funcs}
        new_vars = []
        for key in func_list.keys():
            temp_vars = [v for v in self.all_vars if v.output_type in input_types[key]]
            for v in temp_vars:
                new_vars += [node_cons(key, self.date, v)]
                self.filter_loss(new_vars)
        return new_vars

    def combine_funcs(self, max_funcs, n_inputs=1):
        if n_inputs == 1:
            node_cons = unary_node_cons
            func_list = {k: v for k, v in unary_functions.items() if k in self.unary_funcs}
        elif n_inputs == 2:
            node_cons = BinaryNode
            func_list = {k: v for k, v in binary_functions.items() if k in self.binary_funcs}
        else:
            raise NotImplementedError("n_inputs must be 1 or 2")
        max_index = len(func_list) * len(self.all_vars) ** n_inputs
        n_funcs = min(max_index, max_funcs)
        new_vars = []
        for _ in range(n_funcs):
            key = np.random.choice(np.array(list(func_list.keys())))
            temp_vars = [v for v in self.all_vars if
                         v.output_type in input_types[key] and v.complexity < self.max_complexity]
            if len(temp_vars) == 0:
                continue
            xs = self.sample_by_inverse_complexity(n_inputs, temp_vars)
            new_vars += [node_cons(key, self.date, *xs)]
            self.filter_loss(new_vars)
        return new_vars

    def filter_loss(self, new_vars):
        loss = new_vars[-1].compute_loss(self.loss_fn, self.train_y)
        if np.isnan(loss) or np.isinf(loss):
            new_vars.pop()

    def sample_by_inverse_complexity(self, n_inputs, temp_vars):
        # have changed this to correlation
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
        r = ((-min_losses).argsort().argsort() + 1).astype(float)
        r -= complexity[self.n_base:]
        # Probability of survival is proportional to rank
        # and scaled to target pop_size
        p = r / np.sum(r) * (pop_size - 20)
        # Best 20 are kept with 100% prob
        p[r > np.max(r) - 20] = 1.0
        base_ind = np.full((self.n_base,), True)
        rem_ind = np.random.random((p.shape)) < p
        # oldest = np.argmin([x.birth for x in self.all_vars])
        full_index = np.concatenate((base_ind, rem_ind))
        # full_index[oldest] = False
        return full_index

    def sindySTLS(self, threshold=0.1, thresh_inc=0.05, max_thresh=100, n_param_target=5):
        opt = STLSQ(threshold)
        opt.fit(self.get_library(), self.train_y.cpu())
        idx = opt.ind_.squeeze()
        coef = opt.coef_.squeeze()[idx]
        return coef, idx, None, None

    def STLS(self, threshold=0.01, thresh_inc=0.05, max_thresh=100, n_param_target=5):
        d = self.get_library()
        dtmp = d
        idx = np.arange(d.shape[-1])
        # clf = LinearRegression(fit_intercept=False)
        clf = Ridge(fit_intercept=False)
        min_loss = np.inf
        last_coef = None
        repeated = 0
        while dtmp.shape[-1] > n_param_target and threshold < max_thresh and repeated < 2:
            n_cmp = -1
            threshold += thresh_inc
            while n_cmp != dtmp.shape[-1]:
                clf.fit(dtmp, self.train_y.cpu())
                coef = clf.coef_
                y_hat = clf.predict(dtmp)
                loss = self.loss_fn(torch.Tensor(y_hat).to(device=self.train_y.device), self.train_y)
                # print(f"Loss: {loss:.4f}")
                if loss < min_loss:
                    best_idx = idx
                    best_coef = coef
                    min_loss = loss
                flt = np.abs(coef) > threshold
                if np.all(flt):
                    repeated += 1
                else:
                    repeated = 0
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

    def get_library(self):
        # Lots of scalars would ruin the ridge regression
        self.lib_vars = [v for i, v in enumerate(self.all_vars) if i < self.n_base or type(v) is not ScalarNode]
        dictionary = [v.value for v in self.lib_vars]
        d = torch.cat(dictionary, dim=1).cpu()
        return d

    def compute_stls(self, keep_overfit=False):
        stls = self.STLS
        if self.use_sindy:
            stls = self.sindySTLS
        coef, idx, best_coef, best_idx = stls()
        if len(coef) == 0:
            print("STLS found nothing")
            return
        final_var, new_vars = self.convert_to_func(coef, idx)
        print(final_var.get_name())
        final_var.compute_loss(self.loss_fn, self.train_y)
        # self.all_vars += new_vars + [final_var]
        self.stls_vars += [final_var]
        if keep_overfit:
            final_var, new_vars = self.convert_to_func(best_coef, best_idx)
            print(final_var.get_name())
            final_var.compute_loss(self.loss_fn, self.train_y)
            self.all_vars += new_vars + [final_var]

    def convert_to_func(self, coef, idx):
        scalars = [ScalarNode(self.base_vars[0].x, f"{c:.2f}", c) for c in coef]
        v = np.array(self.lib_vars)[idx].tolist()
        new_vars = [BinaryNode("*", self.date, c, f) for c, f in zip(scalars, v)]
        final_var = new_vars[0]
        for v in new_vars[1:]:
            final_var = BinaryNode("+", self.date, final_var, v)
        return final_var, new_vars

    def remove_duplicates(self):
        self.sort_by_complexity()
        all_v = torch.cat([x.value for x in self.all_vars], axis=-1)
        idx = non_duplicate_columns(all_v)
        idx = self.ensure_base(idx)
        self.all_vars = np.array(self.all_vars)[idx].tolist()
        return all_v[:, idx]

    def filter_vars(self):
        all_v = self.remove_duplicates()
        idx = linearly_independent_columns(all_v)
        idx = self.ensure_base(idx)
        return np.array(self.all_vars)[idx].tolist()

    def ensure_base(self, idx):
        return np.unique(idx.tolist() + [i for i in range(self.n_base)])

    def sort_by_complexity(self):
        tmp_vars = [x for i, x in enumerate(self.all_vars) if i >= self.n_base]
        cplxty = [x.complexity for x in tmp_vars]
        self.all_vars = self.all_vars[:self.n_base] + np.array(tmp_vars)[np.argsort(cplxty)].tolist()

    def compute_val_loss(self):
        all_v = torch.cat([x.forward(self.val_x).unsqueeze(-1) for x in self.all_vars + self.stls_vars], axis=-1)
        # assume MSE Loss:
        val_losses = ((all_v.T - self.val_y) ** 2).mean(-1).tolist()

        for l, v in zip(val_losses, self.all_vars + self.stls_vars):
            v.val_loss = l

    def index_vars(self):
        nodes = []
        comps = ["==", "<", ">"]
        for idx in self.x_indices:
            x = self.base_vars[idx]
            vals = x.value.unique()
            for val in vals:
                snode = ScalarNode(x.value, f"{val:.2f}", val)
                nodes += [BinaryNode(comp, self.date, x, snode) for comp in comps]
        return nodes

def run_tree(x, y, pop_size, epochs):
    tree = FunctionTree(x, y, torch.nn.MSELoss())
    tree.train(pop_size=pop_size, epochs=epochs)
    return tree


def linearly_independent_columns(matrix, tol=1e-10):
    matrix = matrix.cpu()
    # Convert input to numpy array if it's not already
    A = np.array(matrix, dtype=float)

    # Compute the SVD of the matrix
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Find the rank of the matrix
    r = np.sum(s > tol)

    # Get the first r columns of V.T
    V_r = Vt[:r, :].T

    # Find the indices of the columns with the largest magnitude in each row of V_r
    independent_cols = np.abs(V_r).argmax(axis=0)

    # Return the matrix with only independent columns
    return np.unique(independent_cols)


def non_duplicate_columns(data):
    if type(data) is torch.Tensor:
        data = data.cpu().numpy()
    ind = np.lexsort(data)
    diff = np.any(data.T[ind[1:]] != data.T[ind[:-1]], axis=1)
    edges = np.where(diff)[0] + 1
    result = np.split(ind, edges)
    keep_ind = np.sort([group[0] for group in result])
    return keep_ind


def imitate_graph_transition(logdir):
    n_envs = 2
    data_size = 1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load policy
    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs)
    nn_agent = symbolic_agent_constructor(policy)
    # generate data
    m_in, m_out, u_in, u_out, vm_in, vm_out, vu_in, vu_out = generate_data(nn_agent, env, int(data_size))

    x = torch.FloatTensor(m_in).to(device=device)
    y = torch.FloatTensor(m_out).to(device=device)

    #
    tree = FunctionTree(x, y, torch.nn.MSELoss(),
                        # unary_funcs=u_funcs,
                        # binary_funcs=b_funcs,
                        max_complexity=20,
                        validation_ratio=0.2,
                        use_sindy=False,
                        x_indices=[1, 3])

    tree.train(pop_size=200, epochs=5, find_split_points=True)

    model = tree.get_best()

    y_hat = model.forward(x).cpu()
    yplot = y.cpu()
    plt.scatter(yplot, yplot)
    plt.scatter(yplot, y_hat)
    plt.show()


if __name__ == "__main__":
    logdir = "logs/train/cartpole/2024-07-21__09-27-36__seed_6033"
    imitate_graph_transition(logdir)
