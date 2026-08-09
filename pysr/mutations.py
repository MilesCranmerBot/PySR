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

    PySR default weight: ``0.0346``.

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
    """Replace an operator with another operator of the same arity.

    PySR default weight: ``0.293``.
    """


@dataclass(frozen=True)
class FeatureMutation(_ParameterlessMutation):
    """Change the feature referenced by a variable node.

    PySR default weight: ``0.1``.
    """


@dataclass(frozen=True)
class SwapOperandsMutation(_ParameterlessMutation):
    """Swap the operands of a binary operator.

    PySR default weight: ``0.198``.
    """


@dataclass(frozen=True)
class AddNodeMutation(_ParameterlessMutation):
    """Append a node to the expression.

    PySR default weight: ``2.47``.
    """


@dataclass(frozen=True)
class InsertNodeMutation(_ParameterlessMutation):
    """Insert a node above an existing node.

    PySR default weight: ``0.0112``.
    """


@dataclass(frozen=True)
class DeleteNodeMutation(_ParameterlessMutation):
    """Delete a node from the expression.

    PySR default weight: ``0.870``.
    """


@dataclass(frozen=True)
class RotateTreeMutation(_ParameterlessMutation):
    """Rotate a subtree.

    PySR default weight: ``4.26``.
    """


@dataclass(frozen=True)
class BacksolveMutation(AbstractMutation):
    """Fit a replacement expression by backsolving through the expression.

    PySR default weight: ``0.0``.
    """

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
    """Simplify constant parts of the expression.

    PySR default weight: ``0.00209``.
    """


@dataclass(frozen=True)
class RandomizeMutation(_ParameterlessMutation):
    """Replace the expression with a random expression.

    PySR default weight: ``0.000502``.
    """


@dataclass(frozen=True)
class OptimizeMutation(_ParameterlessMutation):
    """Optimize constants as a mutation.

    PySR default weight: ``0.0``.
    """


@dataclass(frozen=True)
class DoNothingMutation(_ParameterlessMutation):
    """Leave the expression unchanged.

    PySR default weight: ``0.273``.
    """
