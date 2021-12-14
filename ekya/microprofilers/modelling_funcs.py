import numpy as np
from scipy.optimize import curve_fit


# ======= Fit wrappers ==========
# Polynomial func
def poly_fit(xp, yp, deg=1):
    z = np.polyfit(xp, yp, deg)
    p = np.poly1d(z)
    return p


# Scipy curve_fit
def scipy_fit(func, xp, yp, sigma=None):
    popt, pcov = curve_fit(func, xp, yp, sigma=sigma, method='dogbox', absolute_sigma=True)
    return lambda x: func(x, *popt)


# ======= Curve Functions ==========
def optimus_curve(x, b0, b1, b2):
    return 1 - (1 / (b0 * x + b1) + b2)


def log_curve(x, a, b, c):
    return a * np.exp(-b * x) + c


# ======= Generators ==========

def base_generator(microprofile_x: np.array,
                   microprofile_y: np.array,
                   **kwargs):
    '''
    Base signature of a fn generator. Must have microprofile_x and microprofile_y as args.
    :param microprofile_x:
    :param microprofile_y:
    :param kwargs:
    :return:
    '''
    return lambda x: 0


def get_simple_optimus_fn(microprofile_x: np.array,
                          microprofile_y: np.array,
                          seed_x: np.array,
                          seed_y: np.array,
                          weight: int):
    '''
    Generates an approximation of the optimus curve. Weighs the seed_x and seed_y points more heavily (according to weight)
    to account for errors in microprofiling
    :param seed_x:
    :param seed_y:
    :param weight: Sets the number of points to be fetched from the seed_curve for use in final curve estimation. Indirectly sets weights for the seed points.
    :param microprofile_x:
    :param microprofile_y:
    :return:
    '''
    # Generate seed curve
    seed_curve = scipy_fit(optimus_curve, seed_x, seed_y)

    # Generate booster points for the seed_curve
    booster_pts_x = np.linspace(min(seed_x), max(seed_x), weight + 2)
    booster_pts_y = seed_curve(booster_pts_x)

    xp = np.concatenate([booster_pts_x, seed_x, microprofile_x])
    yp = np.concatenate([booster_pts_y, seed_y, microprofile_y])
    fn = scipy_fit(optimus_curve, xp, yp)
    return fn


DEFAULT_SCALED_OPTIMUS_ARGS = {
    'end_acc': 0.95,
    'end_epochs': 30,
    'weight': 2,
    'microprofile_expectation_factor': 0.95,
    'upscale_y': False
}
def get_scaled_optimus_fn(microprofile_x: np.array,
                          microprofile_y: np.array,
                          start_acc: float,
                          end_acc: float,
                          end_epochs: float,
                          weight: int,
                          microprofile_expectation_factor: float = 0.95,
                          upscale_y: bool = True):
    '''
    Generates an approximation of the optimus curve and also scales the final accuracy proportional to microprofiling deviation from expected value. Weighs the start_acc and end_acc points more heavily (according to weight).
    to account for errors in microprofiling
    :param upscale_y: Whether to scale up the accuracy end estimates if the microprofiling results are better than expected.
    :param microprofile_expectation_factor: Expected reduction in microprofile accuracies.
    :param start_acc: Starting accuracy
    :param end_acc: End accuracy
    :param end_epochs: The number of epochs to achieve end accuracy
    :param weight: Sets the number of points to be fetched from the seed_curve for use in final curve estimation. Indirectly sets weights for the seed points.
    :param microprofile_x:
    :param microprofile_y:
    :return:
    '''
    # Generate seed curve
    seed_x = np.array([0, end_epochs])
    seed_y = np.array([start_acc, end_acc])
    seed_curve = scipy_fit(optimus_curve, seed_x, seed_y)

    # Calculate the deviation of the microprofiles and generate new curve to fit that.
    microprofile_expected_values = seed_curve(microprofile_x)
    if upscale_y:
        max_factor = 1 / end_acc
    else:
        max_factor = 1
    mean_microprofile_deviation = min(
        np.mean(microprofile_y / (microprofile_expected_values * microprofile_expectation_factor)),
        max_factor)  # Cap deviation at 1
    new_end_acc = end_acc * mean_microprofile_deviation
    new_seed_y = np.array([start_acc, new_end_acc])
    seed_curve = scipy_fit(optimus_curve, seed_x, new_seed_y)

    # Generate booster points for the new_seed_curve
    booster_pts_x = np.linspace(min(seed_x), max(seed_x), weight + 2)
    booster_pts_y = seed_curve(booster_pts_x)

    xp = np.concatenate([booster_pts_x, seed_x, microprofile_x])
    yp = np.concatenate([booster_pts_y, new_seed_y, microprofile_y])
    fn = scipy_fit(optimus_curve, xp, yp)
    return fn


def get_linear_fn(a: float,
                  b: float):
    '''
    Generates a linear curve y=ax+b. Doesn't do fitting.
    '''
    return lambda x: a * x + b


def get_fitted_linear_fn(microprofile_x: np.array,
                         microprofile_y: np.array):
    '''
    Fits a linear curve to the input data
    '''
    return poly_fit(microprofile_x, microprofile_y, deg=1)
