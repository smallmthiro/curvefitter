import csv
import sys
import typing
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

import curvefitter


def main(args: typing.List[str]):
    input_data_file: str = ''
    if 2 > len(args):
        input_data_file = input('Enter the file path(csv) : ')
    else:
        input_data_file = args[1]

    input_data: typing.Optional[np.ndarray] = None
    with open(input_data_file, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        input_data = \
            np.array([[float(elem) for elem in line] for line in csv_reader])

    if input_data.ndim != 2:
        raise ValueError('input_data is not a two-dimensional array.')

    class ModelType(Enum):
        Sigmoid = '0'
        Gompertz = '1'

    model_type_value: str = ''
    if 3 > len(args):
        model_type_value = input('Enter the model type.\n'
                                 + ModelType.Sigmoid.value + ' : ' + ModelType.Sigmoid.name + '\n'
                                 + ModelType.Gompertz.value + ' : ' + ModelType.Gompertz.name + '\n'
                                 + ': ')
    else:
        model_type_value = args[2]

    model_type: typing.Optional[ModelType] = None
    if model_type_value == ModelType.Sigmoid.value:
        model_type = ModelType.Sigmoid
    elif model_type_value == ModelType.Gompertz.value:
        model_type = ModelType.Gompertz
    else:
        raise ValueError('A non-existent model type has been set.')

    x: np.ndarray = input_data[:, 0]
    y: np.ndarray = input_data[:, 1]

    model_func: typing.Optional[curvefitter.IModelFunc] = None
    if model_type == ModelType.Sigmoid:
        model_func = curvefitter.SigmoidModelFunc()
    elif model_type == ModelType.Gompertz:
        model_func = curvefitter.GompertzModelFunc()
    else:
        raise ValueError('A non-existent model type has been set.')

    solver: curvefitter.Solver = curvefitter.Solver(model_func)

    curve_func: curvefitter.ICurveFunc = solver.get_curve_func(x, y)
    estimated_y: np.ndarray = curve_func(x)

    estimate_label: str = ''
    if model_type == ModelType.Sigmoid:
        estimate_label = 'estimate  a:{0[0]:.03f}, b:{0[1]:.03f}, c:{0[2]:.03f}, d:{0[3]:.03f}'.format(
            curve_func.get_beta())
    elif model_type == ModelType.Gompertz:
        estimate_label = 'estimate  a:{0[0]:.03f}, b:{0[1]:.03f}, c:{0[2]:.03f}, u:{0[3]:.03f}, v:{0[3]:.03f}'.format(
            curve_func.get_beta())
    else:
        raise ValueError('A non-existent model type has been set.')

    plt.scatter(x, y, s=5, color='red', alpha=0.7, label='test data')
    plt.plot(x, estimated_y, color='blue', alpha=0.7, label=estimate_label)
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.show()


if __name__ == '__main__':
    args: typing.List[str] = sys.argv
    main(args)
