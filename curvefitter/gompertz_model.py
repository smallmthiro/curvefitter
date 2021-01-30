import math
import typing

import numpy as np
import sympy

import curvefitter.levenberg_marquardt_algo as lmalgo


class GompertzResidualCalculator(lmalgo.IResidualCalculator):
    def __init__(self,
                 exec_expr: typing.Callable[
                     [np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray]) -> None:
        self.__exec_expr: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray] = exec_expr

    def calc(self, x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError('x is not a one-dimensional array.')
        if y.ndim != 1:
            raise ValueError('y is not a one-dimensional array.')
        if x.shape != y.shape:
            raise ValueError('The number of data for x and y do not match.')
        if beta.shape != (5,):
            raise ValueError('Incorrect number of beta parameters.')

        a: float = beta[0]
        b: float = beta[1]
        c: float = beta[2]
        u: float = beta[3]
        v: float = beta[4]

        ret_value = self.__exec_expr(x, y, a, b, c, u, v)
        if not isinstance(ret_value, np.ndarray):
            return np.full(x.shape[0], ret_value)
        else:
            return ret_value


class GompertzJacobiMatElemCalculator(lmalgo.IJacobiMatElemCalculator):
    def __init__(self,
                 exec_expr: typing.Callable[
                     [np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray]) -> None:
        self.__exec_expr: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray] = exec_expr

    def calc(self, x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError('x is not a one-dimensional array.')
        if y.ndim != 1:
            raise ValueError('y is not a one-dimensional array.')
        if x.shape != y.shape:
            raise ValueError('The number of data for x and y do not match.')
        if beta.shape != (5,):
            raise ValueError('Incorrect number of beta parameters.')

        a: float = beta[0]
        b: float = beta[1]
        c: float = beta[2]
        u: float = beta[3]
        v: float = beta[4]

        ret_value = self.__exec_expr(x, y, a, b, c, u, v)
        if not isinstance(ret_value, np.ndarray):
            return np.full(x.shape[0], ret_value)
        else:
            return ret_value


class GompertzInitialEstimateBetaCalculator(lmalgo.IInitialEstimateBetaCalculator):
    def __init__(self,
                 exec_model_slope_expr_c: typing.Callable[
                     [float, float, float], float]) -> None:
        self.__exec_model_slope_expr_c: typing.Callable[
            [float, float, float], float] = exec_model_slope_expr_c

    def calc(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError('x is not a one-dimensional array.')
        if y.ndim != 1:
            raise ValueError('y is not a one-dimensional array.')
        if x.shape != y.shape:
            raise ValueError('The number of data for x and y do not match.')

        y_max_index: int = np.argmax(y)
        y_min_index: int = np.argmin(y)

        a: float = 0.0
        v: float = 0.0
        dy: float = 0.0
        if x[y_max_index] > x[y_min_index]:
            v = y[y_min_index]
            a = y[y_max_index] - v
            dy = a / (x[y_max_index] - x[y_min_index])
        elif x[y_max_index] < x[y_min_index]:
            v = y[y_max_index]
            a = y[y_min_index] - v
            dy = a / (x[y_min_index] - x[y_max_index])
        else:
            raise ValueError('The max and min values ​​of y are the same.')

        b: float = math.log(2)
        c: float = self.__exec_model_slope_expr_c(dy, a, b)

        u: float = x[np.argmin(
            np.abs(y - ((y[y_max_index] - y[y_min_index]) / 2.0 + y[y_min_index])))]

        return np.array([a, b, c, u, v])


class GompertzCurveFunc(lmalgo.ICurveFunc):
    def __init__(self,
                 exec_expr: typing.Callable[
                     [np.ndarray, float, float, float, float, float], np.ndarray],
                 beta: np.ndarray) -> None:
        self.__exec_expr: typing.Callable[
            [np.ndarray, float, float, float, float, float], np.ndarray] = exec_expr
        self.__beta: np.ndarray = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError('x is not a one-dimensional array.')
        if self.__beta.shape != (5,):
            raise ValueError('Incorrect number of beta parameters.')

        a: float = self.__beta[0]
        b: float = self.__beta[1]
        c: float = self.__beta[2]
        u: float = self.__beta[3]
        v: float = self.__beta[4]
        return self.__exec_expr(x, a, b, c, u, v)

    def get_beta(self) -> np.ndarray:
        return self.__beta


class GompertzModelFunc(lmalgo.IModelFunc):
    def __init__(self) -> None:
        x: sympy.Symbol = sympy.Symbol('x')
        y: sympy.Symbol = sympy.Symbol('y')
        a: sympy.Symbol = sympy.Symbol('a')
        b: sympy.Symbol = sympy.Symbol('b')
        c: sympy.Symbol = sympy.Symbol('c')
        u: sympy.Symbol = sympy.Symbol('u')
        v: sympy.Symbol = sympy.Symbol('v')

        model_expr: sympy.Expr = \
            a * sympy.exp(-b * sympy.exp(-c * (x - u))) + v
        residual_expr: sympy.Expr = y - model_expr
        jacobi_mat_elem_expr_a: sympy.Expr = sympy.diff(residual_expr, a)
        jacobi_mat_elem_expr_b: sympy.Expr = sympy.diff(residual_expr, b)
        jacobi_mat_elem_expr_c: sympy.Expr = sympy.diff(residual_expr, c)
        jacobi_mat_elem_expr_u: sympy.Expr = sympy.diff(residual_expr, u)
        jacobi_mat_elem_expr_v: sympy.Expr = sympy.diff(residual_expr, v)
        dy: sympy.Symbol = sympy.Symbol('dy')
        model_slope_expr_c: sympy.Expr = sympy.solve(
            (dy - sympy.diff(model_expr, x)).subs([(x, 0), (u, 0), (v, 0)]), c)[0]

        self.__exec_model_expr: typing.Callable[
            [np.ndarray, float, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, a, b, c, u, v), model_expr, 'numpy')
        self.__exec_residual_expr: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, y, a, b, c, u, v), residual_expr, 'numpy')
        self.__exec_jacobi_mat_elem_expr_a: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, y, a, b, c, u, v), jacobi_mat_elem_expr_a, 'numpy')
        self.__exec_jacobi_mat_elem_expr_b: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, y, a, b, c, u, v), jacobi_mat_elem_expr_b, 'numpy')
        self.__exec_jacobi_mat_elem_expr_c: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, y, a, b, c, u, v), jacobi_mat_elem_expr_c, 'numpy')
        self.__exec_jacobi_mat_elem_expr_u: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, y, a, b, c, u, v), jacobi_mat_elem_expr_u, 'numpy')
        self.__exec_jacobi_mat_elem_expr_v: typing.Callable[
            [np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray] = sympy.lambdify(
                (x, y, a, b, c, u, v), jacobi_mat_elem_expr_v, 'numpy')
        self.__exec_model_slope_expr_c: typing.Callable[
            [float, float, float], float] = sympy.lambdify(
                (dy, a, b), model_slope_expr_c, 'numpy')

    def __call__(self, x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError('x is not a one-dimensional array.')
        if beta.shape != (5,):
            raise ValueError('Incorrect number of beta parameters.')

        a: float = beta[0]
        b: float = beta[1]
        c: float = beta[2]
        u: float = beta[3]
        v: float = beta[4]
        return self.__exec_model_expr(x, a, b, c, u, v)

    def get_residual_calculator(self) -> lmalgo.IResidualCalculator:
        return GompertzResidualCalculator(self.__exec_residual_expr)

    def get_jacobi_mat_elem_calculator(self) -> typing.Tuple[lmalgo.IJacobiMatElemCalculator, ...]:
        return \
            GompertzJacobiMatElemCalculator(
                self.__exec_jacobi_mat_elem_expr_a), \
            GompertzJacobiMatElemCalculator(
                self.__exec_jacobi_mat_elem_expr_b), \
            GompertzJacobiMatElemCalculator(
                self.__exec_jacobi_mat_elem_expr_c), \
            GompertzJacobiMatElemCalculator(
                self.__exec_jacobi_mat_elem_expr_u), \
            GompertzJacobiMatElemCalculator(
                self.__exec_jacobi_mat_elem_expr_v)

    def get_initial_estimate_beta_calculator(self) -> lmalgo.IInitialEstimateBetaCalculator:
        return GompertzInitialEstimateBetaCalculator(self.__exec_model_slope_expr_c)

    def create_curve_func(self, beta: np.ndarray) -> lmalgo.ICurveFunc:
        return GompertzCurveFunc(self.__exec_model_expr, beta)
