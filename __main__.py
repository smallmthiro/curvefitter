import csv
import sys
import typing

import matplotlib.pyplot as plt
import numpy as np

import curvefitter


def main(args: typing.List[str]):
    input_data_file: str = ''
    if 2 > len(args):
        input_data_file = input('Enter the file path(csv) : ')
    else:
        input_data_file = args[1]

    input_data: np.ndarray = None
    with open(input_data_file, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        input_data = \
            np.array([[float(elem) for elem in line] for line in csv_reader])

    if input_data.ndim != 2:
        raise ValueError('input_data is not a two-dimensional array.')

    x: np.ndarray = input_data[:, 0]
    y: np.ndarray = input_data[:, 1]

    model_func: curvefitter.IModelFunc = curvefitter.SigmoidModelFunc()
    solver: curvefitter.Solver = curvefitter.Solver(model_func)

    curve_func: curvefitter.ICurveFunc = solver.get_curve_func(x, y)
    estimated_y: np.ndarray = curve_func(x)

    estimate_label: str = 'estimate  a:{0[0]:.03f}, b:{0[1]:.03f}, c:{0[2]:.03f}, d:{0[3]:.03f}'.format(
        curve_func.get_beta())

    plt.scatter(x, y, s=5, color='red', alpha=0.7, label='test data')
    plt.plot(x, estimated_y, color='blue', alpha=0.7, label=estimate_label)
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.show()


if __name__ == '__main__':
    args: typing.List[str] = sys.argv
    main(args)
