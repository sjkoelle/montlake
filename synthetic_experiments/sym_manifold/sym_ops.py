from __future__ import annotations

from sympy import Symbol, FunctionClass, Expr, Matrix
from sympy import lambdify, solve, simplify

import numpy as np

from networkx import DiGraph
from networkx import topological_generations

from typing import Union, Optional, Iterable, Callable, Sequence, Any, Generator
from collections import deque
from itertools import chain, product, combinations
from functools import reduce as func_reduce

from sym_types import Numeric, SympyNumeric, SymbolInputs, ExpressionInput, ExpressionInputs, DomainInput
from sym_constants import inv_rel_tol, inv_abs_tol, inv_default_num_sel_vals
from sym_utils import parse_symbol, parse_expr, broadcast_batch_input, parse_domain


class Node(object):

    def __init__(self,
                 shape: Optional[tuple[int, ...]] = (),
                 numpy_value: Optional[np.ndarray] = None,
                 sympy_value: Optional[np.ndarray] = None,
                 inv_values: Optional[list[NonConstant]] = None,
                 diff_value: Optional[Node] = None,
                 can_invert: Optional[bool] = None,
                 can_diff: Optional[bool] = None):

        self.shape = shape

        self.numpy_value = numpy_value
        self.sympy_value = sympy_value
        self.inv_values = inv_values
        self.diff_value = diff_value

        self.can_invert = can_invert
        self.can_diff = can_diff

    def __hash__(self): return super(Node, self).__hash__()

    def __eq__(self, other) -> bool: return self is other

    def __add__(self, other: Node) -> Union[Constant, Output]: return Add.add(self, other)

    def __sub__(self, other: Node) -> Union[Constant, Output]: return Sub.sub(self, other)

    def __mul__(self, other: Node) -> Union[Constant, Output]: return Mul.mul(self, other)

    def __truediv__(self, other: Node) -> Union[Constant, Output]: return Div.div(self, other)

    def __pow__(self, other: Node) -> Union[Constant, Output]: return Pow.pow(self, other)

    def __matmul__(self, other: Node) -> Union[Constant, Output]: return MatMul.matmul(self, other)

    @property
    def ndim(self) -> int: return len(self.shape)

    @property
    def is_zero(self) -> bool: return False


class Constant(Node):

    def __init__(self,
                 value: Numeric):

        value = np.array(value)
        if value.size == 1:
            value = value.reshape(())

        super(Constant, self).__init__(shape=value.shape,
                                       numpy_value=value,
                                       sympy_value=value.astype(object),
                                       can_invert=False,
                                       can_diff=True)

    @property
    def is_zero(self) -> bool: return not self.numpy_value.any()


class Var(Node):

    def __init__(self,
                 name: str,
                 shape: Optional[tuple[int, ...]] = ()):

        self.name = name
        super(Var, self).__init__(shape=shape, can_invert=True, can_diff=len(shape) < 2)

    def set_sympy_value(self, value: Optional[SympyNumeric] = None) -> None:

        if value is None:
            self.sympy_value = parse_symbol(self.name, self.shape)
        else:
            self.sympy_value = np.broadcast_to(value, self.shape).astype(object)

    def set_numpy_value(self, value: Numeric,
                        batch_input: Optional[bool] = False) -> None:

        value = np.array(value)
        shape = (value.shape[0], ) + self.shape if batch_input else self.shape

        self.numpy_value = np.broadcast_to(value, shape)

    def diff(self,
             clear: Optional[bool] = True) -> Constant:

        assert self.can_diff

        if len(self.shape):
            self.diff_value = Constant(np.identity(self.shape[0], dtype=float))
        else:
            self.diff_value = Constant(np.array([[1.0]], dtype=float))

        value = self.diff_value

        if clear:
            self.clear()

        return value

    def clear(self) -> None:

        self.numpy_value = None
        self.sympy_value = None

        self.inv_values = None
        self.diff_value = None


class Output(Node):

    def __init__(self,
                 inp_vars: list[Var],
                 parent_func: Function,
                 shape: tuple[int, ...],
                 **kwargs):

        self.inp_vars = inp_vars
        self.parent_func = parent_func
        self.kwargs = kwargs

        super(Output, self).__init__(shape=shape,
                                     can_invert=parent_func.can_invert,
                                     can_diff=parent_func.can_diff)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.eval_numpy(*args, **kwargs, clear=True)

    @property
    def inp_var(self) -> Var:
        assert self.num_vars == 1
        return self.inp_vars[0]

    @property
    def inp_shape(self) -> tuple[int, ...]:
        return self.inp_var.shape

    @property
    def num_vars(self) -> int:
        return len(self.inp_vars)

    def topological_order(self, inverse_order=False) -> Generator[Function, None, None]:

        inv_graph = DiGraph()
        inv_graph.add_node(0, node=self)

        added_nodes = {self: 0}

        next_nodes = deque()
        next_nodes.append(self)

        while next_nodes:

            cur_node = next_nodes.popleft()
            cur_idx = added_nodes[cur_node]

            children = (cur_node.parent_func, ) if isinstance(cur_node, Output) else cur_node.inputs

            for child in children:

                if isinstance(child, Atom) or (inverse_order and not child.can_invert):
                    continue

                if child not in added_nodes:

                    inv_graph.add_node(len(added_nodes), node=child)

                    next_nodes.append(child)
                    added_nodes[child] = len(added_nodes)

                inv_graph.add_edge(cur_idx, added_nodes[child])

        topological_order = []
        for n_idx in chain.from_iterable(topological_generations(inv_graph)):
            node = inv_graph.nodes[n_idx]["node"]
            if isinstance(node, Function):
                topological_order.append(node)

        if inverse_order:
            return iter(topological_order)
        return reversed(topological_order)

    def eval_sympy(self,
                   inputs: Optional[SympyNumeric, dict[Var, SympyNumeric]] = None,
                   clear: Optional[bool] = True) -> Union[np.ndarray, tuple[np.ndarray, ...]]:

        top_node = False

        if inputs is not None:

            if not isinstance(inputs, dict):

                assert self.num_vars == 1
                inputs = {self.inp_var: inputs}

            for var in self.inp_vars:
                var.set_sympy_value(inputs.get(var, None))

            top_node = True

        self.parent_func.eval_sympy()

        value = self.sympy_value

        if top_node and clear:
            self.clear()

        return value

    def eval_numpy(self,
                   inputs: Optional[Numeric, dict[Var, Numeric]] = None,
                   batch_input: Optional[bool] = False,
                   clear: Optional[bool] = True) -> Union[np.ndarray, tuple[np.ndarray, ...]]:

        top_node = False

        if inputs is not None:

            if not isinstance(inputs, dict):

                assert self.num_vars == 1
                inputs = {self.inp_var: inputs}

            for var in self.inp_vars:
                var.set_numpy_value(inputs[var], batch_input)

            top_node = True

        self.parent_func.eval_numpy(batch_input)

        value = self.numpy_value

        if top_node and clear:
            self.clear()

        return value

    def invert(self,
               input_name: str,
               selection_values: Optional[DomainInput] = None,
               num_selection_values: Optional[int] = inv_default_num_sel_vals) -> Union[list[Output], Output]:

        assert self.can_invert

        if selection_values is not None:

            if not isinstance(selection_values, np.ndarray):

                low, high = parse_domain(selection_values, self.inp_shape)
                selection_values = np.random.uniform(low=low, high=high, size=(num_selection_values, ) + low.shape)

            selection_values = self.eval_numpy(selection_values, batch_input=True, clear=False)

        inv_inp_var = Var(input_name, self.shape)
        if selection_values is not None:
            inv_inp_var.set_numpy_value(selection_values, batch_input=True)
        inv_inp_var.set_sympy_value()

        self.inv_values = [inv_inp_var]

        for func in self.topological_order(inverse_order=True):
            func.invert()

        inverses = [inv_value for inv_value in self.inp_var.inv_values]

        self.clear()

        for inv in inverses:
            inv.clear()

        if len(inverses) == 1:
            return inverses[0]
        return inverses

    def diff(self,
             clear: Optional[bool] = True) -> Node:

        assert self.can_diff

        self.parent_func.diff()
        value = self.diff_value

        if clear:
            self.clear()

        return value

    def compose(self,
                inputs: Union[Node, dict[Var, Node]]) -> Node:

        if inputs is not None:
            if not isinstance(inputs, dict):
                assert self.num_vars == 1
                inputs = {self.inp_var: inputs}

        for func in self.topological_order(inverse_order=False):
            func.compose(inputs)

        if self.numpy_value is not None:
            return Constant(self.numpy_value)
        return self

    def clear(self) -> None:

        self.parent_func.clear()

        self.numpy_value = None
        self.sympy_value = None

        self.inv_values = None
        self.diff_value = None

    def to_custom(self, inp_node: Optional[Node] = None) -> Output:

        assert self.num_vars == 1 and self.ndim <= 2

        if inp_node is None:
            inp_node = self.inp_var

        inp_symbols = parse_symbol("_x", self.inp_shape)
        return Custom.custom(inp_node, self.eval_sympy({self.inp_var: inp_symbols}), inp_symbols, can_invert=False)


NonConstant = Union[Output, Var]
Atom = Union[Var, Constant]


class Function(object):

    def __init__(self,
                 inputs: tuple[Node, ...],
                 can_invert: bool,
                 can_diff: bool):

        self.inputs = inputs

        inp_vars = set(inp for inp in inputs if isinstance(inp, Var))
        for inp in inputs:
            if isinstance(inp, Output):
                inp_vars = inp_vars.union(inp.inp_vars)
        inp_vars = list(inp_vars)

        self.can_invert = (can_invert and len(inp_vars) == 1 and any(isinstance(inp, NonConstant) and inp.can_invert for inp in inputs))
        self.can_diff = (can_diff and len(inp_vars) == 1 and all(inp.can_diff for inp in inputs))

        self.outputs = tuple(Output(inp_vars=inp_vars, parent_func=self, **kwg)
                             for kwg in self._init_output_kwargs())

    def __hash__(self): return super(Function, self).__hash__()

    def __eq__(self, other) -> bool: return self is other

    @property
    def non_constant_inputs(self) -> Generator[NonConstant, None, None]:
        return (inp for inp in self.inputs if isinstance(inp, NonConstant))

    @property
    def output_inputs(self) -> Generator[NonConstant, None, None]:
        return (inp for inp in self.inputs if isinstance(inp, Output))

    @property
    def num_inputs(self) -> int: return len(self.inputs)

    @property
    def num_outputs(self) -> int: return len(self.outputs)

    def inp_numpy_values(self, non_constant_only=False) -> Generator[np.ndarray, None, None]:
        if non_constant_only:
            return (inp.numpy_value for inp in self.non_constant_inputs)
        return (inp.numpy_value for inp in self.inputs)

    def inp_sympy_values(self, non_constant_only=False) -> Generator[Node, None, None]:
        if non_constant_only:
            return (inp.sympy_value for inp in self.non_constant_inputs)
        return (inp.sympy_value for inp in self.inputs)

    def inp_diff_values(self) -> Generator[np.ndarray, None, None]:
        return (None if (isinstance(inp, Constant) or inp.diff_value.is_zero) else inp.diff_value
                for inp in self.inputs)

    def out_inv_values(self) -> Generator[list[NonConstant], None, None]:
        return ((out.inv_values or []) for out in self.outputs)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        raise NotImplementedError

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        raise NotImplementedError

    def numpy_function(self, batch_input: Optional[bool] = False) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        return self._numpy_function(tuple(self.inp_numpy_values()), batch_input=batch_input, **vars(self))

    def sympy_function(self) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        return self._sympy_function(tuple(self.inp_sympy_values()), **vars(self))

    def eval_sympy(self) -> None:

        for inp in self.output_inputs:
            if inp.sympy_value is None:
                inp.eval_sympy()

        out_values = self.sympy_function()

        if not isinstance(out_values, tuple):
            out_values = (out_values, )

        for out, out_value in zip(self.outputs, out_values):
            out.sympy_value = out_value

    def eval_numpy(self, batch_input: Optional[bool] = False) -> None:

        for inp in self.output_inputs:
            if inp.numpy_value is None:
                inp.eval_numpy(batch_input=batch_input)

        out_values = self.numpy_function(batch_input)

        if not isinstance(out_values, tuple):
            out_values = (out_values, )

        for out, out_value in zip(self.outputs, out_values):
            out.numpy_value = out_value

    def clear(self) -> None:
        for inp in self.non_constant_inputs:
            inp.clear()

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:
        raise NotImplementedError

    def invert(self) -> None:

        for inv_value_comb in product(*self.out_inv_values()):

            for potential_inv_outs in self.inverse_function(inv_value_comb):

                valid_inv = True

                for inp_val, inv_out in zip(self.inp_numpy_values(non_constant_only=True), potential_inv_outs):

                    if inp_val is not None:
                        if not np.allclose(inp_val, inv_out.eval_numpy(batch_input=True, clear=False),
                                           atol=inv_abs_tol, rtol=inv_rel_tol):
                            valid_inv = False
                            break

                if not valid_inv:
                    continue

                for inp, inv_out in zip(self.non_constant_inputs, potential_inv_outs):

                    inv_out.eval_sympy(clear=False)

                    if inp.inv_values is None:

                        inp.inv_values = [inv_out]
                        continue

                    has_inv = False

                    for inv_value in inp.inv_values:

                        has_inv = True

                        for ex1, ex2 in zip(inv_value.sympy_value.flat, inv_out.sympy_value.flat):
                            if simplify(ex1 - ex2) != 0:
                                has_inv = False
                                break

                        if has_inv:
                            break

                    if not has_inv:
                        inp.inv_values.append(inv_out)

    def diff_function(self) -> list[list[Union[None, Callable]]]:
        raise NotImplementedError

    def diff(self) -> None:

        for inp in self.non_constant_inputs:
            if inp.diff_value is None:
                inp.diff(clear=False)

        for out, out_diffs_wrt_inp in zip(self.outputs, self.diff_function()):

            out_diffs_wrt_var = []

            for inp_diff_wrt_var, out_diff_wrt_inp in zip(self.inp_diff_values(), out_diffs_wrt_inp):
                if inp_diff_wrt_var is not None and out_diff_wrt_inp is not None:
                    out_diffs_wrt_var.append(out_diff_wrt_inp(inp_diff_wrt_var))

            if len(out_diffs_wrt_var) == 1:
                out.diff_value = out_diffs_wrt_var[0]
            elif len(out_diffs_wrt_var) > 1:
                out.diff_value = Reduce.reduce(out_diffs_wrt_var)
            else:
                out.diff_value = Constant(np.broadcast_to(0.0, out.shape))

    def compose(self,
                inputs: dict[Var, Node]) -> None:

        self.inputs = [inputs.get(inp, inp) if (inp.numpy_value is None or isinstance(inp, Constant))
                       else Constant(inp.numpy_value) for inp in self.inputs]

        if all(isinstance(inp, Constant) for inp in self.inputs):
            self.eval_numpy()

        for out in self.outputs:

            new_inp_vars = set()
            if out.numpy_value is None:

                for var, node in inputs.items():

                    try:

                        out.inp_vars.remove(var)

                        if isinstance(node, Var):
                            new_inp_vars.add(node)
                        elif isinstance(node, Output):
                            new_inp_vars = new_inp_vars.union(node.inp_vars)
                        else:
                            raise AssertionError

                    except ValueError:
                        pass

                out.inp_vars = list(new_inp_vars)

    @classmethod
    def create_output(cls,
                      inputs: tuple[Node, ...],
                      *args,
                      **kwargs) -> Union[Output, Constant, tuple[Union[Output, Constant], ...]]:

        if all((isinstance(inp, Constant) for inp in inputs)):

            constant_vals = cls._numpy_function(tuple(inp.numpy_value for inp in inputs), batch_input=False, *args, **kwargs)

            if isinstance(constant_vals, np.ndarray):
                return Constant(constant_vals)
            return tuple(Constant(val) for val in constant_vals)

        else:

            new_func = cls(inputs, *args, **kwargs)

            if new_func.num_outputs == 1:
                return new_func.outputs[0]
            return new_func.outputs


class Reduce(Function):

    def __init__(self,
                 inputs: tuple[Node, ...],
                 reduce_func: Optional[np.ufunc] = np.add):

        self.reduce_func = reduce_func
        self.out_shape = np.broadcast_shapes(*(inp.shape for inp in inputs))

        super(Reduce, self).__init__(inputs=inputs, can_invert=False, can_diff=False)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.out_shape), )

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        if kwargs["batch_input"]:
            inp_vals = broadcast_batch_input(inp_vals, tuple(isinstance(inp, NonConstant) for inp in kwargs["inputs"]))
        return np.array(func_reduce(kwargs["reduce_func"], inp_vals))

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        return np.array(func_reduce(kwargs["reduce_func"], inp_vals))

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:
        raise NotImplementedError

    @classmethod
    def reduce(cls,
               inputs: tuple[Node, ...],
               reduce_func: Optional[np.ufunc] = np.add) -> Union[Constant, Output]:
        return super(Reduce, cls).create_output(inputs=inputs, reduce_func=reduce_func)


class Add(Function):

    def __init__(self, inputs: tuple[Node, Node]):

        self.out_shape = np.broadcast_shapes(*(inp.shape for inp in inputs))
        super(Add, self).__init__(inputs=inputs, can_invert=True, can_diff=False)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.out_shape), )

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        if kwargs["batch_input"]:
            inp_vals = broadcast_batch_input(inp_vals, tuple(isinstance(inp, NonConstant) for inp in kwargs["inputs"]))
        return np.array(inp_vals[0] + inp_vals[1])

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        return np.array(inp_vals[0] + inp_vals[1])

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:

        a, b = self.inputs

        out = Sub.sub(out_comb[0], a) if isinstance(a, Constant) else Sub.sub(out_comb[0], b)

        var = a if isinstance(a, NonConstant) else b
        select_idx = (0, ) * (len(self.out_shape) - len(var.shape))
        if select_idx:
            out = Select.select(out, select_idx)

        return [[out]]

    @classmethod
    def add(cls, a: Node, b: Node) -> Union[Constant, Output]:
        return super(Add, cls).create_output(inputs=(a, b))


class Sub(Function):

    def __init__(self, inputs: tuple[Node, Node]):

        self.out_shape = np.broadcast_shapes(*(inp.shape for inp in inputs))
        super(Sub, self).__init__(inputs=inputs, can_invert=True, can_diff=False)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.out_shape), )

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        if kwargs["batch_input"]:
            inp_vals = broadcast_batch_input(inp_vals, tuple(isinstance(inp, NonConstant) for inp in kwargs["inputs"]))
        return np.array(inp_vals[0] - inp_vals[1])

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        return np.array(inp_vals[0] - inp_vals[1])

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:

        a, b = self.inputs

        out = Sub.sub(a, out_comb[0]) if isinstance(a, Constant) else Add.add(out_comb[0], b)

        var = a if isinstance(a, NonConstant) else b
        select_idx = (0,) * (len(self.out_shape) - len(var.shape))
        if select_idx:
            out = Select.select(out, select_idx)

        return [[out]]

    @classmethod
    def sub(cls, a: Node, b: Node) -> Union[Constant, Output]:
        return super(Sub, cls).create_output(inputs=(a, b))


class Mul(Function):

    def __init__(self, inputs: tuple[Node, Node]):

        self.out_shape = np.broadcast_shapes(*(inp.shape for inp in inputs))
        super(Mul, self).__init__(inputs=inputs, can_invert=True, can_diff=False)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.out_shape),)

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        if kwargs["batch_input"]:
            inp_vals = broadcast_batch_input(inp_vals, tuple(isinstance(inp, NonConstant) for inp in kwargs["inputs"]))
        return np.array(inp_vals[0] * inp_vals[1])

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        return np.array(inp_vals[0] * inp_vals[1])

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:

        a, b = self.inputs

        out = Div.div(out_comb[0], a) if isinstance(a, Constant) else Div.div(out_comb[0], b)

        var = a if isinstance(a, NonConstant) else b
        select_idx = (0,) * (len(self.out_shape) - len(var.shape))
        if select_idx:
            out = Select.select(out, select_idx)

        return [[out]]

    @classmethod
    def mul(cls, a: Node, b: Node) -> Union[Constant, Output]:
        return super(Mul, cls).create_output(inputs=(a, b))


class Div(Function):

    def __init__(self, inputs: tuple[Node, Node]):

        self.out_shape = np.broadcast_shapes(*(inp.shape for inp in inputs))
        super(Div, self).__init__(inputs=inputs, can_invert=True, can_diff=False)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.out_shape),)

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        if kwargs["batch_input"]:
            inp_vals = broadcast_batch_input(inp_vals, tuple(isinstance(inp, NonConstant) for inp in kwargs["inputs"]))
        return np.array(inp_vals[0] / inp_vals[1])

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        return np.array(inp_vals[0] / inp_vals[1])

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:

        a, b = self.inputs

        out = Div.div(a, out_comb[0]) if isinstance(a, Constant) else Mul.mul(out_comb[0], b)

        var = a if isinstance(a, NonConstant) else b
        select_idx = (0,) * (len(self.out_shape) - len(var.shape))
        if select_idx:
            out = Select.select(out, select_idx)

        return [[out]]

    @classmethod
    def div(cls, a: Node, b: Node) -> Union[Constant, Output]:
        return super(Div, cls).create_output(inputs=(a, b))


class Pow(Function):

    def __init__(self, inputs: tuple[Node, Node]):

        self.out_shape = np.broadcast_shapes(*(inp.shape for inp in inputs))
        super(Pow, self).__init__(inputs=inputs, can_invert=isinstance(inputs[1], Constant), can_diff=False)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.out_shape),)

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        if kwargs["batch_input"]:
            inp_vals = broadcast_batch_input(inp_vals, tuple(isinstance(inp, NonConstant) for inp in kwargs["inputs"]))
        return np.array(inp_vals[0] ** inp_vals[1])

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        return np.array(inp_vals[0] ** inp_vals[1])

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:

        a, b = self.inputs

        out = Pow.pow(out_comb[0], Constant(1.0) / b)

        select_idx = (0,) * (len(self.out_shape) - len(a.shape))
        if select_idx:
            out = Select.select(out, select_idx)

        return [[out]]

    @classmethod
    def pow(cls, a: Node, b: Node) -> Union[Constant, Output]:
        return super(Pow, cls).create_output(inputs=(a, b))


class Elementwise(Function):

    def __init__(self,
                 inputs: tuple[Node],
                 func: Union[ExpressionInput, FunctionClass],
                 same_inv_func: Optional[bool] = True,
                 can_invert: Optional[bool] = True):

        self.same_inv_func = same_inv_func
        self.shape = inputs[0].shape

        self.func, self.numpy_func, self.sympy_func, self.inp_symbols = self._parse_func(func,
                                                                                         return_func=True,
                                                                                         return_numpy_func=True,
                                                                                         return_sympy_func=True,
                                                                                         return_inp_symbols=True)

        self.inv_funcs: Optional[Matrix] = None
        self.inv_inp_symbols: Optional[np.ndarray] = None

        super(Elementwise, self).__init__(inputs=inputs, can_invert=can_invert, can_diff=False)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.shape), )

    @staticmethod
    def _parse_func(func: Union[ExpressionInput, FunctionClass],
                    return_func: Optional[bool] = False,
                    return_numpy_func: Optional[bool] = False,
                    return_sympy_func: Optional[bool] = False,
                    return_inp_symbols: Optional[bool] = False) -> tuple[Union[Symbol, Expr, Callable], ...]:

        _x = parse_symbol("_x")

        if isinstance(func, FunctionClass):
            func = func(_x)

        elif isinstance(func, ExpressionInput):

            func = parse_expr(func)

            free_symbols = func.free_symbols
            assert len(free_symbols) == 1

            func = func.subs({free_symbols.pop(): _x})

        else:
            raise ValueError("Cannot parse elementwise func", func)

        return_values = []

        if return_func:
            return_values.append(func)
        if return_numpy_func:
            return_values.append(lambdify([_x], func, modules="numpy"))
        if return_sympy_func:
            return_values.append(np.vectorize(lambdify([_x], func, modules="sympy"), otypes=[object]))
        if return_inp_symbols:
            return_values.append(_x)

        return tuple(return_values)

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:

        numpy_func = kwargs.get("numpy_func", None)
        if numpy_func is None:
            numpy_func = cls._parse_func(kwargs["func"], return_numpy_func=True)[0]

        return np.array(numpy_func(inp_vals[0]))

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:

        sympy_func = kwargs.get("sympy_func", None)
        if sympy_func is None:
            sympy_func = cls._parse_func(kwargs["func"], return_sympy_func=True)[0]

        return np.array(sympy_func(inp_vals[0]))

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:

        if self.inv_funcs is None:

            _y = parse_symbol("_y")

            inv_funcs = solve([self.func - _y], self.inp_symbols, dict=True)
            inv_funcs = [inv_func[self.inp_symbols] for inv_func in inv_funcs]

            if not inv_funcs:
                raise ValueError("Could not invert function", self.func)

            if self.same_inv_func or len(inv_funcs) == 1 or not self.shape:

                self.inv_funcs = inv_funcs
                self.inv_inp_symbols = _y

            else:

                self.inv_inp_symbols = parse_symbol("_y", self.shape)
                mat_reshape = (self.shape[0], 1) if len(self.shape) == 1 else self.shape

                self.inv_funcs = [Matrix([inv_func.subs(_y, inp_sym) for inp_sym, inv_func in
                                          zip(self.inv_inp_symbols.flat, inv_func_comb)]).reshape(*mat_reshape)
                                  for inv_func_comb in product(inv_funcs, repeat=int(np.prod(self.shape)))]

        if self.same_inv_func or len(self.inv_funcs) == 1 or not self.shape:
            return [[Elementwise.elementwise(out_comb[0], inv_func, self.same_inv_func)]
                    for inv_func in self.inv_funcs]
        else:
            return [[Custom.custom(out_comb[0], inv_func, self.inv_inp_symbols)]
                    for inv_func in self.inv_funcs]

    @classmethod
    def elementwise(cls,
                    inp: Node,
                    func: Union[ExpressionInput, FunctionClass],
                    same_inv_func: Optional[bool] = True,
                    can_invert: Optional[bool] = True) -> Union[Constant, Output]:

        return super(Elementwise, cls).create_output(inputs=(inp, ), func=func, can_invert=can_invert,
                                                     same_inv_func=same_inv_func)


class Custom(Function):

    def __init__(self,
                 inputs: tuple[Node],
                 funcs: ExpressionInputs,
                 input_order: Optional[SymbolInputs],
                 can_invert: Optional[bool] = True):

        self.funcs, self.numpy_func, self.sympy_func, self.inp_symbols = (
            self._parse_funcs(funcs, input_order, inputs[0].shape, return_func=True,
                              return_numpy_func=True, return_sympy_func=True, return_inp_symbols=True))

        self.inv_funcs: Optional[Matrix] = None
        self.inv_inp_symbols: Optional[np.ndarray] = None

        self.inp_shape = inputs[0].shape
        self.out_shape = (self.funcs.shape[0], ) if self.funcs.shape[1] == 1 else self.funcs.shape
        self.num_over_constrains = np.prod(self.out_shape) - np.prod(self.inp_shape)

        assert not can_invert or self.num_over_constrains >= 0
        can_diff = len(self.inp_shape) == 1 and len(self.out_shape) == 1

        super(Custom, self).__init__(inputs=inputs, can_invert=can_invert, can_diff=can_diff)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.out_shape), )

    @staticmethod
    def _parse_funcs(funcs: ExpressionInputs,
                     input_order: Optional[SymbolInputs],
                     input_shape: tuple[int, ...],
                     return_func: Optional[bool] = False,
                     return_numpy_func: Optional[bool] = False,
                     return_sympy_func: Optional[bool] = False,
                     return_inp_symbols: Optional[bool] = False) -> tuple[Union[Expr, Callable, np.ndarray], ...]:

        if isinstance(funcs, Matrix):

            assert isinstance(input_order, np.ndarray)
            inp_symbols = input_order

        else:

            inp_symbols = parse_symbol("_x", input_shape)

            if isinstance(funcs, dict):

                exprs = [parse_expr(expr) for expr in funcs.values()]
                out_pos = np.array(list(funcs.keys()))

                funcs = inp_symbols.copy()

                if out_pos.ndim == 1:
                    funcs[out_pos] = exprs
                else:
                    funcs[out_pos[:, 0], out_pos[:, 1]] = exprs

                if input_order is None:
                    input_order = sorted(set().union(*(expr.free_symbols for expr in exprs)),
                                         key=lambda s: s.name)

                funcs = Matrix(funcs)

            else:

                if isinstance(funcs[0], ExpressionInput):
                    funcs = Matrix([parse_expr(func) for func in funcs])
                else:
                    funcs = Matrix([[parse_expr(func) for func in func_row] for func_row in funcs])

                if input_order is None:
                    input_order = sorted(funcs.free_symbols, key=lambda s: s.name)

            if isinstance(input_order, dict):
                subs_dict = {inp_symbol: inp_symbols[inp_pos] for inp_symbol, inp_pos in input_order.items()}
            else:
                input_order = np.array(input_order, dtype=object).reshape(input_shape)
                subs_dict = dict(zip(input_order.flat, inp_symbols.flat))

            funcs = funcs.subs(subs_dict)

        return_values = []

        if return_func:
            return_values.append(funcs)
        if return_numpy_func:
            return_values.append(lambdify(inp_symbols.flatten().tolist(), funcs.flat(), modules="numpy"))
        if return_sympy_func:
            return_values.append(lambdify(inp_symbols.flatten().tolist(), funcs, modules="sympy"))
        if return_inp_symbols:
            return_values.append(inp_symbols)

        return tuple(return_values)

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:

        inp = inp_vals[0]

        numpy_func = kwargs.get("numpy_func", None)
        funcs = kwargs.get("funcs", None)

        if numpy_func is None:
            funcs, numpy_func = cls._parse_funcs(kwargs["funcs"], kwargs["input_order"],
                                                 input_shape=inp.shape,
                                                 return_func=True,
                                                 return_numpy_func=True)

        if kwargs["batch_input"]:

            val = inp.reshape((inp.shape[0], -1)).T
            val = np.stack(np.broadcast_arrays(*numpy_func(*val)))
            val = val.reshape(funcs.shape + (inp.shape[0], ))

        else:
            val = np.array(numpy_func(*inp.flat)).reshape(funcs.shape)

        val = val.squeeze()
        if kwargs["batch_input"]:
            val = np.moveaxis(val, -1, 0)

        return val

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:

        inp = inp_vals[0]

        sympy_func = kwargs.get("sympy_func", None)
        if sympy_func is None:
            sympy_func = cls._parse_funcs(kwargs["funcs"],
                                          kwargs["input_order"],
                                          input_shape=inp[0].shape,
                                          return_sympy_func=True)[0]

        val = np.array(sympy_func(*inp.flat))
        return val.squeeze()

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:

        if self.inv_funcs is None:

            self.inv_inp_symbols = parse_symbol("_y", out_comb[0].shape)
            inp_symbols = self.inp_symbols.flatten().tolist()

            eqs = np.array(self.funcs).reshape(out_comb[0].shape) - self.inv_inp_symbols
            eqs = eqs.flatten()

            num_eqs = len(eqs)

            mat_reshape = (self.inp_shape[0], 1) if len(self.inp_shape) == 1 else self.inp_shape

            self.inv_funcs = []
            for eqs_idx in combinations(range(num_eqs), num_eqs - self.num_over_constrains):
                for new_inv_func in solve(eqs[list(eqs_idx)], inp_symbols, dict=True):
                    if len(new_inv_func) == len(inp_symbols):
                        self.inv_funcs.append(Matrix(list(new_inv_func.values())).reshape(*mat_reshape))

            if not self.inv_funcs:
                raise ValueError("Could not invert function", self.funcs)

        return [[Custom.custom(out_comb[0], inv_func, self.inv_inp_symbols, can_invert=(self.num_over_constrains == 0))]
                for inv_func in self.inv_funcs]

    def diff_function(self) -> list[list[Union[None, Callable]]]:

        self_diff = Custom.custom(self.inputs[0],
                                  Matrix.hstack(*(self.funcs.diff(inp_symbol) for inp_symbol in self.inp_symbols)),
                                  self.inp_symbols, can_invert=False)

        def diff_out_wrt_inp(diff):
            return MatMul.matmul(self_diff, diff)

        return [[diff_out_wrt_inp]]

    @classmethod
    def custom(cls,
               inp: Node,
               funcs: ExpressionInputs,
               input_order: Optional[SymbolInputs] = None,
               can_invert: Optional[bool] = True) -> Union[Constant, Output]:

        return super(Custom, cls).create_output(inputs=(inp, ), funcs=funcs, input_order=input_order,
                                                can_invert=can_invert)


class MatMul(Function):

    def __init__(self,
                 inputs: tuple[Node, Node]):

        A, B = inputs

        assert 1 <= len(A.shape) <= 2
        assert 1 <= len(B.shape) <= 2
        assert A.shape[-1] == B.shape[0]

        can_invert = True
        if not isinstance(A, Constant) and not isinstance(B, Constant):
            can_invert = False
        elif isinstance(A, Constant):
            if len(A.shape) < 2 or A.shape[0] < A.shape[1] or (1.0/np.linalg.cond(A.numpy_value)) < 1e-3:
                can_invert = False
        elif isinstance(B, Constant):
            if len(B.shape) < 2 or B.shape[0] > B.shape[1] or (1.0/np.linalg.cond(B.numpy_value)) < 1e-3:
                can_invert = False

        A_str = "ab" if len(A.shape) == 2 else "b"
        B_str = "bc" if len(B.shape) == 2 else "b"
        C_str = "d" + "".join(sorted(set(A_str).symmetric_difference(B_str)))

        if not isinstance(A, Constant):
            A_str = "d" + A_str
        if not isinstance(B, Constant):
            B_str = "d" + B_str

        self.einsum_str = A_str + "," + B_str + "->" + C_str

        if len(A.shape) == len(B.shape) == 1:
            self.out_shape = ()
        elif len(A.shape) == 2 and len(B.shape) == 1:
            self.out_shape = (A.shape[0], )
        elif len(A.shape) == 1 and len(B.shape) == 2:
            self.out_shape = (B.shape[1], )
        else:
            self.out_shape = (A.shape[0], B.shape[1])

        can_diff = len(A.shape) == 1 or len(B.shape) == 1
        super(MatMul, self).__init__(inputs=inputs, can_invert=can_invert, can_diff=can_diff)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.out_shape), )

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        if kwargs["batch_input"]:
            return np.array(np.einsum(kwargs["einsum_str"], *inp_vals))
        else:
            return np.array(inp_vals[0] @ inp_vals[1])

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        return np.array(inp_vals[0] @ inp_vals[1])

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:

        assert self.can_invert

        A, B = self.inputs

        if isinstance(A, Constant):
            return [[MatMul.matmul(Constant(np.linalg.pinv(A.numpy_value)), out_comb[0])]]
        else:
            return [[MatMul.matmul(out_comb[0], Constant(np.linalg.pinv(B.numpy_value)))]]

    def diff_function(self) -> list[list[Union[None, Callable]]]:

        A, B = self.inputs

        diff_out_wrt_A = None
        diff_out_wrt_B = None

        if not isinstance(A, Constant):

            def diff_out_wrt_A(diff):
                return MatMul.matmul(diff, B)

        if not isinstance(B, Constant):

            def diff_out_wrt_B(diff):
                return MatMul.matmul(A, diff)

        return [[diff_out_wrt_A, diff_out_wrt_B]]

    @classmethod
    def matmul(cls, A: Node, B: Node) -> Union[Constant, Output]:
        return super(MatMul, cls).create_output(inputs=(A, B))


class Select(Function):

    def __init__(self,
                 inputs: tuple[Node],
                 idx: Union[slice, tuple[slice, ...], Numeric]):

        self.idx = idx
        self.out_shape = np.empty(inputs[0].shape)[idx].shape
        super(Select, self).__init__(inputs=inputs, can_invert=False, can_diff=False)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.out_shape), )

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        if kwargs["batch_input"]:
            if isinstance(kwargs['idx'], tuple):
                return np.array(inp_vals[0][(slice(None), ) + kwargs['idx']])
            else:
                return np.array(inp_vals[0][:, kwargs['idx']])
        else:
            return np.array(inp_vals[0][kwargs['idx']])

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        return np.array(inp_vals[0][kwargs['idx']])

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:
        raise NotImplementedError

    @classmethod
    def select(cls, inp: Node, idx: Union[slice, Numeric, tuple[slice, ...]]) -> Union[Constant, Output]:
        return super(Select, cls).create_output(inputs=(inp, ), idx=idx)


class Reshape(Function):

    def __init__(self, inputs: tuple[Node], out_shape: Union[int, tuple[int, ...]]):

        if out_shape == -1:
            out_shape = (int(np.prod(inputs[0].shape)),)

        self.out_shape = out_shape
        self.inp_shape = inputs[0].shape

        super(Reshape, self).__init__(inputs=inputs, can_invert=True, can_diff=False)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.out_shape), )

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        if kwargs["batch_input"]:
            return np.reshape(inp_vals[0], (-1, ) + kwargs["out_shape"])
        else:
            return np.reshape(inp_vals[0], kwargs["out_shape"])

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        return np.reshape(inp_vals[0], kwargs["out_shape"])

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:
        return [[Reshape.reshape(out_comb[0], self.inp_shape)]]

    @classmethod
    def reshape(cls, inp: Node, out_shape: Union[int, tuple[int, ...]]) -> Union[Constant, Output]:
        return super(Reshape, cls).create_output(inputs=(inp, ), out_shape=out_shape)


class Concat(Function):

    def __init__(self,
                 inputs: tuple[Node, ...],
                 axis: Optional[int] = 0) -> Union[Constant, Output]:

        shape = list(inputs[0].shape)
        split_points = []

        for inp in inputs[1:]:
            assert len(inp.shape) == len(shape)
            for i, s in enumerate(inp.shape):
                if i == axis:
                    split_points.append(shape[i])
                    shape[i] += s
                else:
                    assert shape[i] == s

        self.axis = axis
        self.split_points = tuple(split_points)
        self.out_shape = tuple(shape)

        super(Concat, self).__init__(inputs=inputs, can_invert=True, can_diff=False)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:
        return (dict(shape=self.out_shape),)

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        return np.concatenate(inp_vals, axis=kwargs["axis"] + int(kwargs["batch_input"]))

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> np.ndarray:
        return np.concatenate(inp_vals, axis=kwargs["axis"])

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:
        return [Split.split(out_comb[0], self.split_points, self.axis)]

    @classmethod
    def concat(cls,
               inputs: Sequence[Node],
               axis: Optional[int] = 0) -> Union[Constant, Output]:

        return super(Concat, cls).create_output(inputs=inputs, axis=axis)


class Split(Function):

    def __init__(self,
                 inputs: tuple[Node],
                 split_points: Sequence[int],
                 axis: Optional[int] = 0):

        self.split_points = list(split_points)
        self.axis = axis

        sp_with_ends = [0] + self.split_points + [inputs[0].shape[self.axis]]
        num_outputs = len(self.split_points) + 1

        self.out_shapes = [list(inputs[0].shape) for _ in range(num_outputs)]
        for i in range(num_outputs):
            self.out_shapes[i][self.axis] = sp_with_ends[i + 1] - sp_with_ends[i]
        self.out_shapes = tuple(tuple(out_shape) for out_shape in self.out_shapes)

        super(Split, self).__init__(inputs=inputs, can_invert=True, can_diff=False)

    def _init_output_kwargs(self) -> Iterable[dict[str, Any]]:

        return (dict(shape=out_shape) for out_shape in self.out_shapes)

    @classmethod
    def _numpy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> tuple[np.ndarray]:
        return tuple(np.split(inp_vals[0],
                              indices_or_sections=kwargs["split_points"],
                              axis=kwargs["axis"] + int(kwargs["batch_input"])))

    @classmethod
    def _sympy_function(cls, inp_vals: tuple[np.ndarray, ...], *args, **kwargs) -> tuple[np.ndarray]:
        return tuple(np.split(inp_vals[0],
                              indices_or_sections=kwargs["split_points"],
                              axis=kwargs["axis"]))

    def inverse_function(self, out_comb: tuple[NonConstant, ...]) -> list[list[Output]]:
        return [[Concat.concat(out_comb, self.axis)]]

    @classmethod
    def split(cls,
              inp: Node,
              split_points: Sequence[int],
              axis: Optional[int] = 0) -> tuple[Union[Output, Constant], ...]:

        return super(Split, cls).create_output(inputs=(inp, ), split_points=split_points, axis=axis)


add = Add.add
sub = Sub.sub
div = Div.div
mul = Mul.mul
pow = Pow.pow
elementwise = Elementwise.elementwise
custom = Custom.custom
select = Select.select
reshape = Reshape.reshape
concat = Concat.concat
split = Split.split
reduce = Reduce.reduce


if __name__ == "__main__":

    from sympy import sin, cos, log, exp, atan, acos

    x, y, z = Var("x"), Var("y", shape=(3,)), Var("z", shape=(6, 3))
    c1, c2, c3 = Constant(np.random.rand()), Constant([np.random.rand() for _ in range(3)]), Constant(
        np.random.rand(6, 3))

    for op in (add, sub, div, mul, pow):

        for val1, val2 in combinations((x, y, z, c1, c2, c3), 2):

            op_val = op(val1, val2)

            if isinstance(val1, Constant) and isinstance(val2, Constant):
                assert isinstance(op_val, Constant) and op_val.numpy_value is not None and op_val.sympy_value is not None
            else:
                assert isinstance(op_val, Output) and op_val.numpy_value is None and op_val.sympy_value is None

            if not isinstance(op_val, Constant):

                print(op_val.eval_numpy({x: 3.0, y: [3.0, 2.0, 4.0], z: np.ones((6, 3))}))
                print(op_val.eval_sympy({y: Symbol("t") + 2.0}))
                print(op_val.eval_numpy({x: np.random.random((10, )),
                                         y: np.random.random((10, 3)),
                                         z: np.random.random((10, 6, 3))}, batch_input=True))
                print(op_val.eval_sympy(dict()))

                assert op_val.numpy_value is None and op_val.sympy_value is None

            assert x.sympy_value is None and x.numpy_value is None
            assert y.sympy_value is None and y.numpy_value is None
            assert z.sympy_value is None and z.numpy_value is None

            assert c1.sympy_value is not None and c1.numpy_value is not None
            assert c2.sympy_value is not None and c2.numpy_value is not None
            assert c3.sympy_value is not None and c3.numpy_value is not None

    for op in (add, sub, div, mul, pow):
        for val1, val2 in product((x, y, z), (c1, c2, c3)):
            for v1, v2 in ((val1, val2), (val2, val1)):

                op_val = op(val1, val2)

                if op is pow:
                    sel_vals = (0.0, 2.0)
                else:
                    sel_vals = (-2.0, 2.0)

                inv_val = op_val.invert(input_name="o", selection_values=sel_vals)

                op_var = next(iter(op_val.inp_vars))
                inv_var = next(iter(inv_val.inp_vars))

                inp_array = np.random.random(op_var.shape)
                inv_out_array = inv_val.eval_numpy(op_val.eval_numpy(inp_array))

                assert inp_array.shape == inv_out_array.shape
                assert np.allclose(inp_array, inv_out_array)

                assert inv_val.numpy_value is None and inv_val.sympy_value is None
                assert not inv_val.inv_values

                assert inv_var.numpy_value is None and inv_var.sympy_value is None
                assert not inv_var.inv_values

                assert op_val.numpy_value is None and op_val.sympy_value is None
                assert not op_val.inv_values

                assert isinstance(v1, Constant) or (v1.numpy_value is None and v1.sympy_value is None and not v1.inv_values)
                assert isinstance(v2, Constant) or (v2.numpy_value is None and v2.sympy_value is None and not v2.inv_values)

    for fun in (sin, cos, log, exp, atan, acos):
        for val1 in (x, y, z, c1, c2, c3):

            op_val = elementwise(val1, fun, same_inv_func=(len(val1.shape) == 2), can_invert=True)

            print(fun, val1, val1.shape)

            if isinstance(val1, Constant):

                assert isinstance(op_val, Constant)

            else:

                print(op_val.eval_numpy({x: np.random.random((10,)),
                                         y: np.random.random((10, 3)),
                                         z: np.random.random((10, 6, 3))}, batch_input=True).shape)

                op_var = next(iter(op_val.inp_vars))

                inp_val = np.random.random(op_var.shape)

                inv_val = op_val.invert(input_name='o',
                                        selection_values=np.stack([np.random.random(op_var.shape) for _ in range(10)]))
                op_var = next(iter(op_val.inp_vars))
                inv_var = next(iter(inv_val.inp_vars))

                inp_array = np.random.random(op_var.shape)
                inv_out_array = inv_val.eval_numpy(op_val.eval_numpy(inp_array))

                assert inp_array.shape == inv_out_array.shape
                assert np.allclose(inp_array, inv_out_array)

                assert inv_val.numpy_value is None and inv_val.sympy_value is None
                assert not inv_val.inv_values

                assert inv_var.numpy_value is None and inv_var.sympy_value is None
                assert not inv_var.inv_values

                assert op_val.numpy_value is None and op_val.sympy_value is None
                assert not op_val.inv_values

                assert val1.numpy_value is None and val1.sympy_value is None and not val1.inv_values

    print((z @ c2).eval_sympy(dict()), (c3 @ y).eval_sympy(dict()), (c3 @ c2).numpy_value)

    for _ in range(10):

        M = Constant(np.random.random((6, 3)))
        m = Constant(np.random.random((6, )))

        op_val = M @ y + m

        inv_val = op_val.invert(input_name="o")

        op_var = next(iter(op_val.inp_vars))
        inv_var = next(iter(inv_val.inp_vars))

        inp_array = np.random.random(op_var.shape)
        inv_out_array = inv_val.eval_numpy(op_val.eval_numpy(inp_array))

        assert inp_array.shape == inv_out_array.shape
        assert np.allclose(inp_array, inv_out_array)

        assert inv_val.numpy_value is None and inv_val.sympy_value is None
        assert not inv_val.inv_values

        assert inv_var.numpy_value is None and inv_var.sympy_value is None
        assert not inv_var.inv_values

        assert op_val.numpy_value is None and op_val.sympy_value is None
        assert not op_val.inv_values

        assert y.numpy_value is None and y.sympy_value is None and not y.inv_values

    print(reshape(z, -1).eval_sympy(dict()), reshape(c3, -1).numpy_value)

    reshape_inv = reshape(z, -1).invert(input_name="t")
    print(next(iter(reshape_inv.inp_vars)).shape, reshape_inv.shape, reshape_inv.eval_sympy(dict()))

    print(tuple(out_val.shape for out_val in split(z, [1, 4])))
    print(tuple(out_val.eval_sympy(dict()) for out_val in split(z, [1, 2], axis=1)))
    print(tuple(out_val.numpy_value for out_val in split(c3, [1, 4])))

    out_val = concat(split(z, [1, 4]))
    print(out_val.eval_sympy(dict()))
    print(out_val.invert(input_name="v").eval_sympy(dict()))

    print(reduce((x, y, z)).eval_sympy(dict()))
    print(reduce((x, y, z)).eval_numpy({x: np.random.random((10,)),
                                        y: np.random.random((10, 3)),
                                        z: np.random.random((10, 6, 3))}, batch_input=True).shape)
    print(reduce((c1, c2, c3)).numpy_value)

    M1 = Var(name="M1", shape=(3, 6))
    M1_const = Constant(np.random.random((3, 3)))

    op_val = M1_const @ M1

    print(op_val.eval_numpy({M1: np.random.random((10, 3, 6))}, batch_input=True).shape)

    inv_val = op_val.invert("y")
    inv_var = next(iter(inv_val.inp_vars))

    eval_val = np.random.random(M1.shape)

    assert np.allclose(inv_val.eval_numpy({inv_var: op_val.eval_numpy({M1: eval_val})}), eval_val)

    M1 = Var(name="M1", shape=(3,))
    M1_const = Constant(np.random.random((6, 3)))

    op_val = M1_const @ M1
    print(op_val.diff().numpy_value, M1_const.numpy_value)
    assert np.allclose(op_val.diff().numpy_value, M1_const.numpy_value)

    print(op_val.eval_numpy({M1: np.random.random((10, 3))}, batch_input=True).shape)

    inv_val = op_val.invert("y")
    inv_var = next(iter(inv_val.inp_vars))

    eval_val = np.random.random(M1.shape)

    assert np.allclose(inv_val.eval_numpy({inv_var: op_val.eval_numpy({M1: eval_val})}), eval_val)

    M1 = Var(name="M1", shape=(3,))
    M1_const = Constant(np.random.random((3, 6)))

    op_val = M1 @ M1_const
    assert np.allclose(op_val.diff().numpy_value, M1_const.numpy_value)

    print(op_val.eval_numpy({M1: np.random.random((10, 3))}, batch_input=True).shape)

    inv_val = op_val.invert("y")
    inv_var = next(iter(inv_val.inp_vars))

    assert np.allclose(inv_val.eval_numpy({inv_var: op_val.eval_numpy({M1: eval_val})}), eval_val)

    M1 = Var(name="M1", shape=(4, 3))
    M1_const = Constant(np.random.random((3, 6)))

    op_val = M1 @ M1_const

    print(op_val.eval_numpy({M1: np.random.random((10, 4, 3))}, batch_input=True).shape)

    inv_val = op_val.invert("y")
    inv_var = next(iter(inv_val.inp_vars))

    assert np.allclose(inv_val.eval_numpy({inv_var: op_val.eval_numpy({M1: eval_val})}), eval_val)

    w = Var("w", shape=(2, 2))
    w_const = Constant(np.random.random((2, 2)))
    print(concat((w, w)).eval_sympy(dict()))
    print(concat((w_const, w_const)).numpy_value)
    print(split(w, split_points=[1])[0].eval_sympy(dict()))
    print(split(w_const, split_points=[1])[0].numpy_value)

    print(custom(y, ["sin(x_0)", "cos(x_1)", "exp(x_2)"]).parent_func.funcs)
    print(custom(y, ["sin(x_0)", "cos(x_1)", "exp(x_2)"], input_order=("x_2", "x_0", "x_1")).parent_func.funcs)
    print(custom(y, ["sin(x_0)", "cos(x_1)", "exp(x_2)"], input_order={"x_2": 0, "x_0": 1, "x_1": 2}).parent_func.funcs)

    print(custom(y, {0: "sin(x_0 + 2x_2)", 1: "x_1", 2: "x_0 * x_2"}).parent_func.funcs)
    print(custom(y, {0: "sin(x_0 + 2x_2)", 2: "x_0 * x_2"}, input_order=("x_0", "x_1", "x_2")).parent_func.funcs)

    cus_func = custom(w, ["sin(x_0)", "cos(x_1)", "exp(x_2)", "asin(x_3)"])
    print(cus_func.eval_sympy(dict()))
    print(cus_func.eval_numpy({w: np.random.random((2, 2))}))
    print(cus_func.eval_numpy({w: np.random.random((10, 2, 2))}, batch_input=True))

    print(cus_func.parent_func.funcs)
    print(cus_func.shape)

    cus_func = custom(w, [["sin(x_0) + x_2", "cos(x_1) - x_1"], ["exp(x_2)", "asin(x_3)"]],
                      input_order={"x_0": (1, 1), "x_1": (1, 0), "x_2": (0, 1), "x_3": (0, 0)})
    print(cus_func.eval_sympy(dict()))
    print(cus_func.eval_numpy({w: np.random.random((2, 2))}))
    print(cus_func.eval_numpy({w: np.random.random((10, 2, 2))}, batch_input=True))
    print(cus_func.parent_func.funcs)
    print(cus_func.shape)

    cus_func = custom(w_const, {(0, 0): "x_0 + x_1", (0, 1): "2 * x_3", (1, 1): "sin(x_2) + cos(x_3)"})
    print(cus_func.numpy_value)
    print(cus_func.shape)

    cus_func = custom(w, ["sin(x_0)", "cos(x_1)", "exp(x_2)", "acos(x_3)"])
    inv_val = cus_func.invert(input_name="o", selection_values=((0, 0, 0, 0), (1, 1, 1, 1)))

    for _ in range(10):

        op_var = next(iter(cus_func.inp_vars))
        inv_var = next(iter(inv_val.inp_vars))

        inp_array = np.random.random(op_var.shape)
        inv_out_array = inv_val.eval_numpy(cus_func.eval_numpy(inp_array))

        assert inp_array.shape == inv_out_array.shape
        assert np.allclose(inp_array, inv_out_array)

        assert inv_val.numpy_value is None and inv_val.sympy_value is None
        assert not inv_val.inv_values

        assert inv_var.numpy_value is None and inv_var.sympy_value is None
        assert not inv_var.inv_values

        assert cus_func.numpy_value is None and cus_func.sympy_value is None
        assert not cus_func.inv_values

        assert w.numpy_value is None and w.sympy_value is None and not w.inv_values

    w = Var("w", shape=(2, ))
    cus_func = custom(w, ["sin(x_0)", "log(x_1) - x_0", "exp(x_0 + 2 * x_1)", "acos(x_1)"])

    for inv_val in cus_func.invert(input_name="o"):
        print(inv_val.eval_sympy(dict()))

    w = Var("w", shape=(2,))
    c1 = Constant(np.zeros((2, )))
    cus_func = custom(concat([w, c1]), ["sin(x_0 + x_2)", "cos(x_1 + x_3)", "exp(x_2)", "acos(x_3)"])

    for inv_val in cus_func.invert(input_name="o"):
        print(inv_val.eval_sympy(dict()))

    v1 = Var(name="v", shape=(3, ))
    c1 = Constant(np.random.random((3, )))

    op_val = v1 @ c1
    assert np.allclose(op_val.diff().numpy_value, c1.numpy_value)

    v1 = Var(name="v", shape=(3,))
    M1 = Constant(np.random.random((4, 3)))
    f = custom(M1 @ v1, ["sin(x_0 + x_2)", "cos(x_1 + x_3)", "exp(x_2)", "asin(x_3)", "x_1 ** 2 + x_0"])

    print(f.diff().eval_sympy(dict()))
    print(f.to_custom().diff().eval_sympy(dict()))

    print(f.compose({v1: c1}))
    print(f, f.numpy_value)

    v1 = Var(name="v", shape=(3,))
    M1 = Constant(np.random.random((4, 3)))
    f = custom(M1 @ v1, ["sin(x_0 + x_2)", "cos(x_1 + x_3)", "exp(x_2)", "asin(x_3)", "x_1 ** 2 + x_0"])

    v2 = Var(name="v", shape=(4,))
    M2 = Constant(np.random.random((3, 4)))
    g = custom(M2 @ v2, ["log(x_0)", "exp(x_1)", "tan(x_2)"])

    h = f.compose({v1: g})
    print(h.eval_sympy(dict()))

    v2 = Var(name="v", shape=(4,))
    M2 = Constant(np.random.random((3, 4)))
    g = custom(M2 @ v2, ["log(x_0)", "exp(x_1)", "tan(x_2)"])

    M1 = Constant(np.random.random((4, 3)))
    f = custom(M1 @ g, ["sin(x_0 + x_2)", "cos(x_1 + x_3)", "exp(x_2)", "asin(x_3)", "x_1 ** 2 + x_0"])

    print(h.eval_sympy(dict()))
