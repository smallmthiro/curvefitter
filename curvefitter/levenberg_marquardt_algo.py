import abc
import typing

import numpy as np


class IResidualCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calc(self, x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        pass


class IJacobiMatElemCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calc(self, x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        pass


class IInitialEstimateBetaCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calc(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass


class ICurveFunc(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_beta(self) -> np.ndarray:
        pass


class IModelFunc(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_residual_calculator(self) -> IResidualCalculator:
        pass

    @abc.abstractmethod
    def get_jacobi_mat_elem_calculator(self) -> typing.Tuple[IJacobiMatElemCalculator, ...]:
        pass

    @abc.abstractmethod
    def get_initial_estimate_beta_calculator(self) -> IInitialEstimateBetaCalculator:
        pass

    @abc.abstractmethod
    def create_curve_func(self, beta: np.ndarray) -> ICurveFunc:
        pass


class Solver():
    def __init__(self, model_func: IModelFunc) -> None:
        self.__model_func: IModelFunc = model_func
        self.__residual_calculator: IResidualCalculator = \
            model_func.get_residual_calculator()
        self.__jacobi_mat_elem_calculator: typing.Tuple[IJacobiMatElemCalculator, ...] = \
            model_func.get_jacobi_mat_elem_calculator()
        self.__initial_estimate_beta_calculator: IInitialEstimateBetaCalculator = \
            model_func.get_initial_estimate_beta_calculator()

    def get_curve_func(self, x: np.ndarray, y: np.ndarray,
                       lambda_: float = 0.001, nu: float = 1.1,
                       rss_convergence: float = 0.000_001, max_iteration: int = 5000) -> ICurveFunc:
        if x.ndim != 1:
            raise ValueError('x is not a one-dimensional array.')
        if y.ndim != 1:
            raise ValueError('y is not a one-dimensional array.')
        if x.shape != y.shape:
            raise ValueError('The number of data for x and y do not match.')
        if len(self.__jacobi_mat_elem_calculator) > x.shape[0] or \
                len(self.__jacobi_mat_elem_calculator) > y.shape[0]:
            raise ValueError('The number of x and y data is not enough.')

        if lambda_ <= 0.:
            raise ValueError('lambda is 0 or less.')
        if nu <= 1.:
            raise ValueError('nu is 1 or less.')

        if rss_convergence <= 0.:
            raise ValueError('rss_convergence is 0 or less.')
        if max_iteration <= 0:
            raise ValueError('max_iteration is 0 or less.')

        beta: np.ndarray = self.__initial_estimate_beta_calculator.calc(x, y)
        if beta.ndim != 1:
            raise ValueError('beta is not a one-dimensional array.')
        if len(self.__jacobi_mat_elem_calculator) > beta.shape[0]:
            raise ValueError('The number of beta parameters is not enough.')

        is_converge: bool = False

        def calc_rss(beta: np.ndarray) -> float:
            return np.sum(np.square(self.__residual_calculator.calc(x, y, beta)))

        rss: float = calc_rss(beta)

        for _ in range(max_iteration):
            jacobi_mat_t: np.ndarray = None
            for elem_calculator in self.__jacobi_mat_elem_calculator:
                if jacobi_mat_t is None:
                    jacobi_mat_t = elem_calculator.calc(x, y, beta)
                else:
                    jacobi_mat_t = np.vstack(
                        (jacobi_mat_t, elem_calculator.calc(x, y, beta)))

            jacobi_mat: np.ndarray = jacobi_mat_t.T
            approx_hesse_mat: np.ndarray = np.matmul(jacobi_mat_t, jacobi_mat)
            residual: np.ndarray = self.__residual_calculator.calc(x, y, beta)

            def calc_next_beta(lambda_: float) -> np.ndarray:
                return beta - np.matmul(np.matmul(np.linalg.pinv(
                    approx_hesse_mat + (lambda_ * np.identity(beta.shape[0]))), jacobi_mat_t), residual)

            for k in range(max_iteration):
                beta_lambda_div_nu: np.ndarray = calc_next_beta(lambda_ / nu)
                rss_lambda_div_nu: float = calc_rss(beta_lambda_div_nu)

                if rss_lambda_div_nu <= rss:
                    if rss - rss_lambda_div_nu < rss_convergence:
                        is_converge = True

                    lambda_ = lambda_ / nu
                    beta = beta_lambda_div_nu
                    rss = rss_lambda_div_nu
                    break
                else:
                    beta_lambda: np.ndarray = calc_next_beta(lambda_)
                    rss_lambda: float = calc_rss(beta_lambda)

                    if rss_lambda <= rss:
                        if rss - rss_lambda < rss_convergence:
                            is_converge = True

                        beta = beta_lambda
                        rss = rss_lambda
                        break
                    else:
                        lambda_ = lambda_ * (nu ** (k + 2))

            if is_converge == True:
                break

        return self.__model_func.create_curve_func(beta)
