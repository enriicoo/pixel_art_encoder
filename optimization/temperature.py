from dataclasses import dataclass
from math import exp, log
from typing import Iterator


@dataclass
class TemperatureParameters:
    """Parameters controlling the temperature schedule."""
    initial_accept_ratio: float = 0.4  # Target initial worse-move acceptance rate
    final_accept_ratio: float = 0.01  # Target final worse-move acceptance rate
    alpha: float = 0.95  # Cooling rate (0.9 to 0.99 typical)
    min_iterations: int = 100  # Minimum iterations per temperature
    max_iterations: int = 1000  # Maximum iterations per temperature
    iterations_multiplier: int = 4  # Scale iterations with problem size


class TemperatureSchedule:
    """
    Manages temperature progression for simulated annealing.
    Implements exponential cooling schedule with adaptive iteration counts.
    """

    def __init__(self, params: TemperatureParameters = None):
        """Initialize with optional custom parameters."""
        self.params = params or TemperatureParameters()

    def calculate_start_temp(self, initial_cost: float,
                             avg_cost_delta: float) -> float:
        """
        Calculate starting temperature to achieve desired initial acceptance ratio.

        Args:
            initial_cost: Cost of initial solution
            avg_cost_delta: Average cost change from sample moves

        Returns:
            Starting temperature
        """
        # Use acceptance probability formula to solve for T:
        # P(accept) = exp(-delta/T)
        # T = -delta / ln(P(accept))
        target_prob = self.params.initial_accept_ratio
        abs_delta = abs(avg_cost_delta)

        if abs_delta == 0:
            # If no cost changes observed, use fraction of initial cost
            return initial_cost * 0.01

        return -abs_delta / log(target_prob)

    def calculate_end_temp(self, start_temp: float,
                           min_cost_delta: float) -> float:
        """
        Calculate ending temperature based on minimum meaningful cost change.

        Args:
            start_temp: Starting temperature
            min_cost_delta: Smallest meaningful cost change

        Returns:
            Ending temperature
        """
        target_prob = self.params.final_accept_ratio
        return -min_cost_delta / log(target_prob)

    def calculate_iterations(self, problem_size: int) -> int:
        """
        Calculate iterations per temperature based on problem size.

        Args:
            problem_size: Size of the problem (e.g., number of pixels)

        Returns:
            Number of iterations to run at each temperature
        """
        # Scale iterations with problem size, bounded by min/max
        iterations = problem_size * self.params.iterations_multiplier
        return max(self.params.min_iterations,
                   min(iterations, self.params.max_iterations))

    def generate_schedule(self, start_temp: float,
                          end_temp: float) -> Iterator[float]:
        """
        Generate sequence of temperatures from start to end.

        Args:
            start_temp: Starting temperature
            end_temp: Ending temperature

        Yields:
            Sequence of temperatures
        """
        current_temp = start_temp
        while current_temp > end_temp:
            yield current_temp
            current_temp *= self.params.alpha

    def acceptance_probability(self, cost_delta: float,
                               temperature: float) -> float:
        """
        Calculate probability of accepting a move.

        Args:
            cost_delta: Change in cost (positive means worse)
            temperature: Current temperature

        Returns:
            Probability of accepting the move [0-1]
        """
        if cost_delta <= 0:  # Better solution
            return 1.0
        return exp(-cost_delta / temperature)