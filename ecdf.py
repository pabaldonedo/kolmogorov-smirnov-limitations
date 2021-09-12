import numpy as np
from typing import Tuple


class EmpiricalCDF:
    """Empirical Cumulative Density Function (CDF)"""

    def __init__(self, samples: np.ndarray) -> None:
        """Initialize class.

        :param samples: samples to generate the empirical CDF from.
        """
        self.x = np.sort(np.unique(samples))
        self.y = 1 / samples.size * np.array([np.sum(samples <= x) for x in self.x])
        self.size = samples.size
        self.evaluate = np.frompyfunc(self._evaluate, 1, 1)

    def staircase_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate x and y vectors for staircase shape.

        For the staircase shape of empirical CDF, ith sample x axis must be repeated twice
        and y axes must be CDF[n-1] and CDF[n]. In addition, one sample prior to first and after last
        sample are added for fully complete the range 0-1.

        :return: x and y coordinates of staircase plot of empirical CDF.
        """
        cdf_x = np.hstack((self.x[0] * (1 - np.sign(self.x[0])*.01),
                           np.repeat(self.x, 2),
                           self.x[-1] * (1 + np.sign(self.x[-1])*.01)))
        cdf_y = np.hstack((0, 0, np.repeat(self.y, 2)[:-1], 1))
        return cdf_x, cdf_y

    def _evaluate(self, x: float) -> float:
        """Evaluate single input value.

        :param x: value to compute CDF value for.
        :return: Cumulative Density Function (CDF) at x.
        """
        if x < np.min(self.x):
            return 0

        if x > np.max(self.x):
            return 1

        return self.y[np.where(self.x <= x)[0][-1]]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Vectorized version of _evaluate, i.e. compute the CDF values at x positions."""
        return self.evaluate(x)

    def __eq__(self, other) -> bool:
        return np.all(self.x == other.x) & np.all(self.y == other.y)

    def __len__(self) -> float:
        return self.size
