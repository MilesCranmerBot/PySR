"""Plugin configurations for :class:`PySRRegressor`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from .julia_import import AnyValue, SymbolicRegression, jl


class AbstractPlugin(ABC):
    """Base class for plugin configurations."""

    @abstractmethod
    def julia_plugin(self) -> AnyValue:
        """Create the corresponding SymbolicRegression.jl plugin."""
        pass  # pragma: no cover


@dataclass(frozen=True)
class SimulatedAnnealingPlugin(AbstractPlugin):
    """Apply simulated annealing during mutation acceptance.

    Enabled by default through ``annealing=True``.
    """

    alpha: float = 0.1

    def julia_plugin(self) -> AnyValue:
        return SymbolicRegression.SimulatedAnnealingPlugin(alpha=self.alpha)


@dataclass(frozen=True)
class AdaptiveParsimonyPlugin(AbstractPlugin):
    """Apply frequency-based parsimony during selection and mutation.

    Enabled by default in PySR through ``use_frequency=True`` and
    ``use_frequency_in_tournament=True``.
    """

    tournament: bool = True
    mutation_acceptance: bool = True

    def julia_plugin(self) -> AnyValue:
        return SymbolicRegression.AdaptiveParsimonyPlugin(
            tournament=self.tournament,
            mutation_acceptance=self.mutation_acceptance,
        )


@dataclass(frozen=True)
class AdaptiveMutationWeightsPlugin(AbstractPlugin):
    """Adapt mutation weights from their observed success rates.

    Disabled by default in PySR.
    Learned runtime state is reinitialized for each call to ``fit``.
    """

    smoothing: float = 0.02
    floor: float = 0.05
    reward: Literal["cost", "loss"] = "cost"

    def julia_plugin(self) -> AnyValue:
        return SymbolicRegression.AdaptiveMutationWeightsPlugin(
            smoothing=self.smoothing,
            floor=self.floor,
            reward=jl.Symbol(self.reward),
        )


@dataclass(frozen=True)
class MutationBurstPlugin(AbstractPlugin):
    """Retry rejected mutations and optionally chain accepted mutations.

    Disabled by default in PySR.
    """

    retry_attempts: int = 4
    compound_probability: float = 0.25
    compound_max_steps: int = 2

    def julia_plugin(self) -> AnyValue:
        return SymbolicRegression.MutationBurstPlugin(
            retry_attempts=self.retry_attempts,
            compound_probability=self.compound_probability,
            compound_max_steps=self.compound_max_steps,
        )
