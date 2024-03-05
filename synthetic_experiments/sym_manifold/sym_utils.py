from sympy import Symbol, Expr, symarray
from sympy.parsing.sympy_parser import T
from sympy.parsing.sympy_parser import parse_expr as sympy_parse_expr

import numpy as np

from typing import Optional, Union, Any
import pickle
import os

from sym_constants import default_symbol_assumptions
from sym_types import SymbolInput, ExpressionInput, DomainInput


def parse_domain(domain_inp: DomainInput,
                 shape: Optional[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:

    if not isinstance(domain_inp, np.ndarray):

        low, high = domain_inp

        low = np.array(low)
        high = np.array(high)

        assert low.shape == high.shape

        if low.ndim == 0 and high.ndim == 0:

            low = np.broadcast_to(low, shape)
            high = np.broadcast_to(high, shape)

        elif low.shape != shape:

            low = low.reshape(shape)
            high = high.reshape(shape)

        return low, high

    else:

        return domain_inp[:, 0], domain_inp[:, 1]


def parse_symbol(symbol: SymbolInput,
                 shape: Optional[tuple[int, ...]] = None,
                 symbol_assumptions: Optional[dict[str, bool]] = default_symbol_assumptions) -> Union[np.ndarray, Symbol]:

    if isinstance(symbol, str):
        if shape is not None:
            return symarray(symbol, shape, **symbol_assumptions)
        return Symbol(symbol, **symbol_assumptions)
    elif isinstance(symbol, Symbol):
        for k, v in symbol_assumptions.items():
            symbol._assumptions[k] = v
        if shape is not None:
            symbol = np.broadcast_to(symbol, shape)
        return symbol
    else:
        raise ValueError("Cannot cast", symbol, "to symbol")


def parse_expr(expr: ExpressionInput) -> Expr:

    if isinstance(expr, (str, Expr)):
        if isinstance(expr, str):
            expr = sympy_parse_expr(expr, transformations=T[:])
    else:
        raise ValueError("Cannot parse", expr, "into an expression")

    return expr


def broadcast_batch_input(inp_vals: tuple[np.ndarray, ...],
                          inp_vals_have_batch: tuple[bool, ...]) -> tuple[np.ndarray, ...]:

    non_batch_shapes = tuple(val.shape[1:] if has_batch else val.shape
                             for val, has_batch in zip(inp_vals, inp_vals_have_batch))
    ndims = tuple(len(shape) for shape in non_batch_shapes)
    max_ndims = max(ndims)

    return tuple(np.expand_dims(val, axis=tuple(range(1, 1 + max_ndims - ndim)))
                 if has_batch and ndim < max_ndims else val
                 for val, has_batch, ndim in zip(inp_vals, inp_vals_have_batch, ndims))


def pickle_save(obj: Any, path: str) -> None:

    if ".pkl" != path[-4:]:
        path = path + ".pkl"

    print("Saving data to", path)

    f = open(path, "wb")
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def pickle_load(path: str) -> Any:

    if ".pkl" != path[-4:]:
        path = path + ".pkl"

    assert os.path.exists(path)

    print("Loading data from", path)

    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()

    return obj
