from typing import List, Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


def fill_std_known(
    mean_temp: np.ndarray,
    std_temp: np.ndarray,
    t: np.ndarray,
    ax: Axes,
    color: str = "C0",
    label: Optional[str] = None,
    alpha: float = 0.2,
) -> Tuple[List[Line2D], Polygon]:
    """
    Plots a mean line with a shaded region representing +/- standard deviation.

    Parameters
    ----------
    mean_temp : np.ndarray
        1D array containing the mean values to plot along the y-axis.
    std_temp : np.ndarray
        1D array containing the standard deviation values corresponding to mean_temp.
        The shaded region will be mean_temp +/- std_temp.
    t : np.ndarray
        1D array containing the time or x-axis values corresponding to mean_temp and std_temp.
        Must have the same length as mean_temp and std_temp.
    ax : matplotlib.axes.Axes
        The Matplotlib Axes object onto which the plot should be drawn.
    color : str, optional
        The color to use for both the mean line and the shaded region.
        Defaults to Matplotlib's first default color ('C0').
    label : str, optional
        The label to apply to the mean line for use in legends. Defaults to None.
    alpha : float, optional
        The alpha transparency level for the shaded standard deviation region.
        Defaults to 0.2.

    Returns
    -------
    Tuple[List[Line2D], Polygon]
        A tuple containing:
        - A list containing the Matplotlib Line2D object for the plotted mean line.
        - The Matplotlib Polygon object representing the shaded standard deviation area.

    Raises
    ------
    ValueError
        If input arrays `mean_temp`, `std_temp`, and `t` do not have the same length.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 100)
    >>> mean_vals = np.sin(t)
    >>> std_vals = np.random.rand(100) * 0.2 + 0.1
    >>> fig, ax = plt.subplots()
    >>> line, shade = fill_std_known(mean_vals, std_vals, t, ax, color='blue', label='Sine Wave')
    >>> ax.legend()
    >>> plt.show()
    """
    if not (len(mean_temp) == len(std_temp) == len(t)):
        raise ValueError(
            "Inputs 'mean_temp', 'std_temp', and 't' must have the same length."
        )
    if not isinstance(ax, Axes):
        raise TypeError("'ax' must be a Matplotlib Axes object.")

    mean_temp = np.asarray(mean_temp)
    std_temp = np.asarray(std_temp)
    t = np.asarray(t)

    shade = ax.fill_between(
        t,
        mean_temp - std_temp,
        mean_temp + std_temp,
        color=color,
        alpha=alpha,
        linewidth=0,
    )

    line = ax.plot(t, mean_temp, label=label, color=color)

    return line, shade
