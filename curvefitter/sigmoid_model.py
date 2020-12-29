import typing

import numpy as np
import sympy

import curvefitter.levenberg_marquardt_algo as lmalgo


class SigmoidResidualCalculator(lmalgo.IResidualCalculator):
    def __init__(self,
                 exec_expr: typing.Callable[
                     [np.ndarray, np.ndarray, float, float, float, float], np.ndarray]) -> None:
        self.__exec_expr: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float], np.ndarray] = exec_expr

    def calc(self, x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError('x is not a one-dimensional array.')
        if y.ndim != 1:
            raise ValueError('y is not a one-dimensional array.')
        if x.shape != y.shape:
            raise ValueError('The number of data for x and y do not match.')
        if beta.shape != (4,):
            raise ValueError('Incorrect number of beta parameters.')

        a: float = beta[0]
        b: float = beta[1]
        c: float = beta[2]
        d: float = beta[3]

        ret_value = self.__exec_expr(x, y, a, b, c, d)
        if not isinstance(ret_value, np.ndarray):
            return np.full(x.shape[0], ret_value)
        else:
            return ret_value


class SigmoidJacobiMatElemCalculator(lmalgo.IJacobiMatElemCalculator):
    def __init__(self,
                 exec_expr: typing.Callable[
                     [np.ndarray, np.ndarray, float, float, float, float], np.ndarray]) -> None:
        self.__exec_expr: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float], np.ndarray] = exec_expr

    def calc(self, x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError('x is not a one-dimensional array.')
        if y.ndim != 1:
            raise ValueError('y is not a one-dimensional array.')
        if x.shape != y.shape:
            raise ValueError('The number of data for x and y do not match.')
        if beta.shape != (4,):
            raise ValueError('Incorrect number of beta parameters.')

        a: float = beta[0]
        b: float = beta[1]
        c: float = beta[2]
        d: float = beta[3]

        ret_value = self.__exec_expr(x, y, a, b, c, d)
        if not isinstance(ret_value, np.ndarray):
            return np.full(x.shape[0], ret_value)
        else:
            return ret_value


class SigmoidInitialEstimateBetaCalculator(lmalgo.IInitialEstimateBetaCalculator):
    def __init__(self,
                 exec_model_slope_expr_a: typing.Callable[
                     [float, float], float]) -> None:
        self.__exec_model_slope_expr_a: typing.Callable[
            [float, float], float] = exec_model_slope_expr_a

    def calc(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError('x is not a one-dimensional array.')
        if y.ndim != 1:
            raise ValueError('y is not a one-dimensional array.')
        if x.shape != y.shape:
            raise ValueError('The number of data for x and y do not match.')

        d: float = np.amin(y)
        b: float = np.amax(y) - d
        c: float = x[np.argmin(np.abs(y - (b / 2.0 + d)))]
        dy: float = b / (x[np.argmax(y)] - x[np.argmin(y)])
        a: float = self.__exec_model_slope_expr_a(dy, b)

        return np.array([a, b, c, d])


class SigmoidCurveFunc(lmalgo.ICurveFunc):
    def __init__(self,
                 exec_expr: typing.Callable[
                     [np.ndarray, float, float, float, float], np.ndarray],
                 beta: np.ndarray) -> None:
        self.__exec_expr: typing.Callable[
            [np.ndarray, float, float, float, float], np.ndarray] = exec_expr
        self.__beta: np.ndarray = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError('x is not a one-dimensional array.')
        if self.__beta.shape != (4,):
            raise ValueError('Incorrect number of beta parameters.')

        a: float = self.__beta[0]
        b: float = self.__beta[1]
        c: float = self.__beta[2]
        d: float = self.__beta[3]
        return self.__exec_expr(x, a, b, c, d)

    def get_beta(self) -> np.ndarray:
        return self.__beta


class SigmoidModelFunc(lmalgo.IModelFunc):
    def __init__(self) -> None:
        x: sympy.Symbol = sympy.Symbol('x')
        y: sympy.Symbol = sympy.Symbol('y')
        a: sympy.Symbol = sympy.Symbol('a')
        b: sympy.Symbol = sympy.Symbol('b')
        c: sympy.Symbol = sympy.Symbol('c')
        d: sympy.Symbol = sympy.Symbol('d')

        model_expr: sympy.Expr = 1 / (1 + sympy.exp(-a * (x - c))) * b + d
        residual_expr: sympy.Expr = y - model_expr
        jacobi_mat_elem_expr_a: sympy.Expr = sympy.diff(residual_expr, a)
        jacobi_mat_elem_expr_b: sympy.Expr = sympy.diff(residual_expr, b)
        jacobi_mat_elem_expr_c: sympy.Expr = sympy.diff(residual_expr, c)
        jacobi_mat_elem_expr_d: sympy.Expr = sympy.diff(residual_expr, d)
        dy: sympy.Symbol = sympy.Symbol('dy')
        model_slope_expr_a: sympy.Expr = sympy.solve(
            (dy - sympy.diff(model_expr, x)).subs([(x, 0), (c, 0), (d, 0)]), a)[0]

        self.__exec_model_expr: typing.Callable[
            [np.ndarray, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, a, b, c, d), model_expr, 'numpy')
        self.__exec_residual_expr: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, y, a, b, c, d), residual_expr, 'numpy')
        self.__exec_jacobi_mat_elem_expr_a: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, y, a, b, c, d), jacobi_mat_elem_expr_a, 'numpy')
        self.__exec_jacobi_mat_elem_expr_b: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, y, a, b, c, d), jacobi_mat_elem_expr_b, 'numpy')
        self.__exec_jacobi_mat_elem_expr_c: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, y, a, b, c, d), jacobi_mat_elem_expr_c, 'numpy')
        self.__exec_jacobi_mat_elem_expr_d: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, y, a, b, c, d), jacobi_mat_elem_expr_d, 'numpy')
        self.__exec_model_slope_expr_a: typing.Callable[
            [float, float], float] = sympy.lambdify(
                (dy, b), model_slope_expr_a, 'numpy')

    def __call__(self, x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError('x is not a one-dimensional array.')
        if beta.shape != (4,):
            raise ValueError('Incorrect number of beta parameters.')

        a: float = beta[0]
        b: float = beta[1]
        c: float = beta[2]
        d: float = beta[3]
        return self.__exec_model_expr(x, a, b, c, d)

    def get_residual_calculator(self) -> lmalgo.IResidualCalculator:
        return SigmoidResidualCalculator(self.__exec_residual_expr)

    def get_jacobi_mat_elem_calculator(self) -> typing.Tuple[lmalgo.IJacobiMatElemCalculator, ...]:
        return \
            SigmoidJacobiMatElemCalculator(
                self.__exec_jacobi_mat_elem_expr_a), \
            SigmoidJacobiMatElemCalculator(
                self.__exec_jacobi_mat_elem_expr_b), \
            SigmoidJacobiMatElemCalculator(
                self.__exec_jacobi_mat_elem_expr_c), \
            SigmoidJacobiMatElemCalculator(
                self.__exec_jacobi_mat_elem_expr_d)

    def get_initial_estimate_beta_calculator(self) -> lmalgo.IInitialEstimateBetaCalculator:
        return SigmoidInitialEstimateBetaCalculator(self.__exec_model_slope_expr_a)

    def create_curve_func(self, beta: np.ndarray) -> lmalgo.ICurveFunc:
        return SigmoidCurveFunc(self.__exec_model_expr, beta)
