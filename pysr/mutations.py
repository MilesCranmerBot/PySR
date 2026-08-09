"""Mutation configurations for :class:`PySRRegressor`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .julia_import import AnyValue, SymbolicRegression


class AbstractMutation(ABC):
    """Base class for mutation configurations."""

    @abstractmethod
    def julia_mutation(self) -> AnyValue:
        """Create the corresponding SymbolicRegression.jl mutation."""
        pass  # pragma: no cover


@dataclass(frozen=True)
class _ParameterlessMutation(AbstractMutation):
    def julia_mutation(self) -> AnyValue:
        return getattr(SymbolicRegression, type(self).__name__)()


@dataclass(frozen=True)
class ConstantMutation(AbstractMutation):
    """Perturb a constant.

    Defaults match SymbolicRegression.jl.
    """

    perturbation_factor: float = 0.086
    probability_negate: float = 0.01

    def julia_mutation(self) -> AnyValue:
        return SymbolicRegression.ConstantMutation(
            perturbation_factor=self.perturbation_factor,
            probability_negate=self.probability_negate,
        )


@dataclass(frozen=True)
class OperatorMutation(_ParameterlessMutation):
    """Replace an operator with another operator of the same arity."""


@dataclass(frozen=True)
class FeatureMutation(_ParameterlessMutation):
    """Change the feature referenced by a variable node."""


@dataclass(frozen=True)
class SwapOperandsMutation(_ParameterlessMutation):
    """Swap the operands of a binary operator."""


@dataclass(frozen=True)
class AddNodeMutation(_ParameterlessMutation):
    """Append a node to the expression."""


@dataclass(frozen=True)
class InsertNodeMutation(_ParameterlessMutation):
    """Insert a node above an existing node."""


@dataclass(frozen=True)
class DeleteNodeMutation(_ParameterlessMutation):
    """Delete a node from the expression."""


@dataclass(frozen=True)
class RotateTreeMutation(_ParameterlessMutation):
    """Rotate a subtree."""


@dataclass(frozen=True)
class BacksolveMutation(AbstractMutation):
    """Fit a replacement expression by backsolving through the expression."""

    max_library_size: int = 500
    lambda_: float = 0.01
    max_iter: int = 10

    def julia_mutation(self) -> AnyValue:
        return SymbolicRegression.BacksolveMutation(
            max_library_size=self.max_library_size,
            max_iter=self.max_iter,
            **{"lambda": self.lambda_},
        )


@dataclass(frozen=True)
class SimplifyMutation(_ParameterlessMutation):
    """Simplify constant parts of the expression."""


@dataclass(frozen=True)
class RandomizeMutation(_ParameterlessMutation):
    """Replace the expression with a random expression."""


@dataclass(frozen=True)
class OptimizeMutation(_ParameterlessMutation):
    """Optimize constants as a mutation."""


@dataclass(frozen=True)
class DoNothingMutation(_ParameterlessMutation):
    """Leave the expression unchanged."""
