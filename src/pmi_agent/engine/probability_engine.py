"""Probability normalization helpers."""


class ProbabilityEngine:
    """Validate and normalize probability-like numeric values."""

    def normalize_probability(self, value: float) -> float:
        """Normalize a probability expressed as 0-1 or 0-100 into 0-1."""

        if 0.0 <= value <= 1.0:
            return value
        if 1.0 < value <= 100.0:
            return value / 100.0
        raise ValueError("Probability must be between 0 and 1, or between 1 and 100 percent.")
