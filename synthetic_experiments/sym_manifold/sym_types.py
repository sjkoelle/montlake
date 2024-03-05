from sympy import Symbol, Expr, Matrix

from numpy import ndarray

from typing import Union, Sequence


Numeric = Union[int, float, Sequence[int], Sequence[float], ndarray]
SympyNumeric = Union[Numeric, Expr]

ExpressionInput = Union[str, Expr]
ExpressionInputs = Union[Matrix, dict[Union[int, tuple[int, int]], ExpressionInput], Sequence[ExpressionInput]]

SymbolInput = Union[str, Symbol]
SymbolInputs = Union[ndarray, dict[SymbolInput, Union[int, tuple[int, int]]], Sequence[SymbolInput]]

DomainInput = Union[ndarray, tuple[float, float], tuple[Sequence[float], Sequence[float]]]
