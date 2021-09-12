import numpy as np
from scipy.stats import kstest, rv_continuous
from typing import Generator, Optional, Tuple


class MontecarloKSTest:
    """Perform Montecarlo simulation for Kolmogorov-Smirnov test with estimated distribution parameters."""

    def __init__(self, montecarlo_samples: int) -> None:
        """Instantiate the MontecarloKSTest class.

        :param montecarlo_samples: number of Montecarlo runs to perform.
        """
        self.montecarlo_samples = int(montecarlo_samples)

    @staticmethod
    def _get_sample_statistic(sample: np.ndarray, dist: rv_continuous) -> float:
        """Compute Dn statistic of the Kolmogorov-Smirnov test after fitting distribution to the sample.

        :param sample: ndarray with the sample under study.
        :param dist: rv_continuous scipy distribution under study.
        :return: Dn statistic of the Kolmogorov-Smirnov test.
        """
        sample_parameters = dist.fit(sample)
        return kstest(sample, dist(*sample_parameters).cdf)[0]

    def _run_montecarlo(self, dist: rv_continuous, sample_parameters: Tuple[float, ...], sample_size: int,
                        get_seed: Optional[Generator] = None) -> np.ndarray:
        """Compute Dn statistics by Montecarlo simulation when fitting distribution parameters from the data.

        Perform the following steps for self.montecarlo_samples times:
            1) Sample `sample_size` samples from distribution `dist` with parameters `sample_parameters`
            2) Fit distribution `dist` to the sample drawn in in step 1)
            3) Compute the Kn statistic of the KS test of sample 1) against distribution 2)

        :param dist: scipy rv_continuous distribution to perform the goodness of it against.
        :param sample_parameters: distribution parameters estimated from the sample under study.
        :param sample_size: sample size.
        :param get_seed: generator to set the seed of each Montecarlo step. Optional.
        :return: array of Dn statistics of each Montecarlo run.
        """
        statistics = []
        if get_seed is None:
            get_seed = (None for _ in range(self.montecarlo_samples))

        sample_dist = dist(*sample_parameters)
        for i, seed in zip(range(self.montecarlo_samples), get_seed):
            simulation = sample_dist.rvs(size=sample_size, random_state=seed)
            statistics.append(self._get_sample_statistic(simulation, dist))

        statistics = np.array(statistics)
        return np.sort(statistics)

    def test(self, sample: np.ndarray, dist: rv_continuous,
             get_seed: Optional[Generator] = None) -> Tuple[float, float]:
        """Compute Dn statistic and p-value via Montecarlo simulation of a KS test with fitted parameters.

        Firstly, compute the Dn statistic of the Kolmogorov-Smirnov test of the `sample` against the
        distribution `dist` with parameters fitted to the sample `sample`.
        Secondly, perform a Montecarlo simulation by repeating the following steps self.montecarlo_samples times:
            1) Sample `sample_size` samples from distribution `dist` with parameters `sample_parameters`
            2) Fit distribution `dist` to the sample drawn in in step 1)
            3) Compute the Kn statistic of the KS test of sample 1) against distribution 2)

        The p-value is computed as the proportion of statistics of the Montecarlo simulation greater than or
        equal to the Dn statistic from `sample`.

        :param sample: data under study.
        :param dist: scipy rv_continuous distribution to perform the goodness of it against.
        :param get_seed: generator to set the seed of each Montecarlo step. Optional.
        :return: Dn statistic of the KS test of the sample and the estimated p-value by Montecarlo simulation.
        """
        sample_size = sample.size
        sample_parameters = dist.fit(sample)

        d0 = self._get_sample_statistic(sample, dist)
        montecarlo_simulation = self._run_montecarlo(dist, sample_parameters, sample_size, get_seed)
        p_value = np.sum(d0 <= montecarlo_simulation) / montecarlo_simulation.size
        return d0, p_value
