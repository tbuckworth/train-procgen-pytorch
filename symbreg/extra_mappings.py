import sympy
import torch


# Allows PyTorch to map Piecewise functions:
def expr_cond_pair(expr, cond):
    if isinstance(cond, torch.Tensor) and not isinstance(expr, torch.Tensor):
        expr = torch.tensor(expr, dtype=cond.dtype, device=cond.device)
    elif isinstance(expr, torch.Tensor) and not isinstance(cond, torch.Tensor):
        cond = torch.tensor(cond, dtype=expr.dtype, device=expr.device)
    else:
        return expr, cond

    # First, make sure expr and cond are same size:
    if expr.shape != cond.shape:
        if len(expr.shape) == 0:
            expr = expr.expand(cond.shape)
        elif len(cond.shape) == 0:
            cond = cond.expand(expr.shape)
        else:
            raise ValueError(
                "expr and cond must have same shape, or one must be a scalar."
            )
    return expr, cond


def piecewise(*expr_conds):
    output = None
    already_used = None
    for expr, cond in expr_conds:
        if not isinstance(cond, torch.Tensor) and not isinstance(
                expr, torch.Tensor
        ):
            # When we just have scalars, have to do this a bit more complicated
            # due to the fact that we need to evaluate on the correct device.
            if output is None:
                already_used = cond
                output = expr if cond else 0.0
            else:
                if not isinstance(output, torch.Tensor):
                    output += expr if cond and not already_used else 0.0
                    already_used = already_used or cond
                else:
                    expr = torch.tensor(
                        expr, dtype=output.dtype, device=output.device
                    ).expand(output.shape)
                    output += torch.where(
                        cond & ~already_used, expr, torch.zeros_like(expr)
                    )
                    already_used = already_used | cond
        else:
            if output is None:
                already_used = cond
                # TODO: found at least two devices:
                # "logs/train/cartpole_continuous/pure-graph/2024-09-08__00-59-06__seed_6033/symbreg/2024-09-19__14-52-58"
                output = torch.where(cond, expr, torch.zeros_like(expr))
            else:
                output += torch.where(
                    cond.bool() & ~already_used, expr, torch.zeros_like(expr)
                )
                already_used = already_used | cond.bool()
    return output


def as_bool(x):
    if isinstance(x, torch.Tensor):
        return x.bool()
    else:
        return bool(x)


def if_then_else(*conds):
    a, b, c = conds
    device = get_device(a, b, c)
    if device is not None:
        a = try_to_bool(a, device)
        b = try_to_bool(b, device)
        c = try_to_bool(c, device)
        return torch.where(a, torch.where(b, True, False), torch.where(c, True, False))
    if a:
        return b
    return c


def get_device(a, b, c):
    try:
        return a.device
    except AttributeError:
        try:
            return b.device
        except AttributeError:
            try:
                return c.device
            except AttributeError:
                return None


def try_to_bool(a, device):
    try:
        a = a.to(bool)
    except AttributeError:
        return torch.BoolTensor([a]).to(device=device)
    return a


def exp1():
    return torch.exp(torch.FloatTensor([1]))


def inf():
    return torch.FloatTensor([torch.inf])


def exp(x):
    return torch.exp(torch.FloatTensor(x))


# TODO: Add test that makes sure tensors are on the same device
def get_extra_torch_mappings():
    return {
        # sympy.Mul: _reduce(torch.mul),
        # sympy.Add: _reduce(torch.add),
        sympy.Heaviside: torch.heaviside,
        sympy.core.numbers.Half: (lambda: 0.5),
        sympy.core.numbers.One: (lambda: 1.0),
        sympy.logic.boolalg.Boolean: as_bool,
        sympy.logic.boolalg.BooleanTrue: (lambda: True),
        sympy.logic.boolalg.BooleanFalse: (lambda: False),
        sympy.functions.elementary.piecewise.ExprCondPair: expr_cond_pair,
        sympy.Piecewise: piecewise,
        sympy.logic.boolalg.ITE: if_then_else,
        sympy.core.numbers.Exp1: exp1,
        sympy.exp: exp,
        sympy.core.numbers.ComplexInfinity: inf,
    }
