import math
import warnings
from random import random
from typing import Optional, Union

from negmas import (
    SAONegotiator,
    Controller,
    Negotiator,
    Issue, MechanismState, ResponseType, outcome_with_utility, utility_range, RandomProposalMixin)


class AspirationMixin:
    """Adds aspiration level calculation. This Mixin MUST be used with a `Negotiator` class."""

    def aspiration_init(
        self,
        max_aspiration: float,
        aspiration_type: Union[str, int, float],
        above_reserved_value=True,
    ):
        """

        Args:
            max_aspiration:
            aspiration_type:
            above_reserved_value:
        """
        if hasattr(self, "add_capabilities"):
            self.add_capabilities({"aspiration": True})
        self.max_aspiration = max_aspiration
        self.aspiration_type = aspiration_type
        self.exponent = 1.0
        if isinstance(aspiration_type, int):
            self.exponent = float(aspiration_type)
        elif isinstance(aspiration_type, float):
            self.exponent = aspiration_type
        elif aspiration_type == "boulware":
            self.exponent = 4.0
        elif aspiration_type == "linear":
            self.exponent = 1.0
        elif aspiration_type == "conceder":
            self.exponent = 0.25
        else:
            raise ValueError(f"Unknown aspiration type {aspiration_type}")
        self.above_reserved = above_reserved_value

    def aspiration(self, t: float) -> float:
        """
        The aspiration level

        Args:
            t: relative time (a number between zero and one)

        Returns:
            aspiration level
        """
        if t is None:
            raise ValueError(
                f"Aspiration negotiators cannot be used in negotiations with no time or #steps limit!!"
            )
        return self.max_aspiration * (1.0 - math.pow(t, self.exponent))



class AspirationNegotiator(SAONegotiator, AspirationMixin):
    """
    Represents a time-based negotiation strategy that is independent of the offers received during the negotiation.

    Args:
        name: The agent name
        ufun:  The utility function to attache with the agent
        max_aspiration: The aspiration level to use for the first offer (or first acceptance decision).
        aspiration_type: The polynomial aspiration curve type. Here you can pass the exponent as a real value or
                         pass a string giving one of the predefined types: linear, conceder, boulware.
        dynamic_ufun: If True, the utility function will be assumed to be changing over time. This is depricated.
        randomize_offer: If True, the agent will propose outcomes with utility >= the current aspiration level not
                         outcomes just above it.
        can_propose: If True, the agent is allowed to propose
        assume_normalized: If True, the ufun will just be assumed to have the range [0, 1] inclusive
        ranking: If True, the aspiration level will not be based on the utility value but the ranking of the outcome
                 within the presorted list. It is only effective when presort is set to True
        ufun_max: The maximum utility value (used only when `presort` is True)
        ufun_min: The minimum utility value (used only when `presort` is True)
        presort: If True, the negotiator will catch a list of outcomes, presort them and only use them for offers
                 and responses. This is much faster then other option for general continuous utility functions
                 but with the obvious problem of only exploring a discrete subset of the issue space (Decided by
                 the `discrete_outcomes` property of the `AgentMechanismInterface` . If the number of outcomes is
                 very large (i.e. > 10000) and discrete, presort will be forced to be True. You can check if
                 presorting is active in realtime by checking the "presorted" attribute.
        tolerance: A tolerance used for sampling of outcomes when `presort` is set to False
        assume_normalized: If true, the negotiator can assume that the ufun is normalized.
        rational_proposal: If `True`, the negotiator will never propose something with a utility value less than its
                        reserved value. If `propose` returned such an outcome, a NO_OFFER will be returned instead.
        owner: The `Agent` that owns the negotiator.
        parent: The parent which should be an `SAOController`

    """

    def __init__(
        self,
        max_aspiration=8.0,
        aspiration_type=5.0,
        dynamic_ufun=True,
        randomize_offer=False,
        can_propose=True,
        assume_normalized=False,
        ranking=False,
        ufun_max=None,
        ufun_min=None,
        presort: bool = True,
        tolerance: float = 0.01,
        **kwargs,
    ):
        self.ordered_outcomes = []
        self.ufun_max = ufun_max
        self.ufun_min = ufun_min
        self.ranking = ranking
        self.tolerance = tolerance
        if assume_normalized:
            self.ufun_max, self.ufun_min = 1.0, 0.0
        super().__init__(
            assume_normalized=assume_normalized, **kwargs,
        )
        self.aspiration_init(
            max_aspiration=max_aspiration, aspiration_type=aspiration_type
        )
        if not dynamic_ufun:
            warnings.warn(
                "dynamic_ufun is deprecated. All Aspiration negotiators assume a dynamic ufun"
            )
        self.randomize_offer = randomize_offer
        self._max_aspiration = self.max_aspiration
        self.best_outcome, self.worst_outcome = None, None
        self.presort = presort
        self.presorted = False
        self.add_capabilities(
            {
                "respond": True,
                "propose": can_propose,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        self.__last_offer_util, self.__last_offer = float("inf"), None
        self.n_outcomes_to_force_presort = 10000
        self.n_trials = 1

    def on_ufun_changed(self):
        super().on_ufun_changed()
        presort = self.presort
        if (
            not presort
            and all(i.is_countable() for i in self._ami.issues)
            and Issue.num_outcomes(self._ami.issues) >= self.n_outcomes_to_force_presort
        ):
            presort = True
        if presort:
            outcomes = self._ami.discrete_outcomes()
            uvals = self.utility_function.eval_all(outcomes)
            uvals_outcomes = [
                (u, o)
                for u, o in zip(uvals, outcomes)
                if u >= self.utility_function.reserved_value
            ]
            self.ordered_outcomes = sorted(
                uvals_outcomes,
                key=lambda x: float(x[0]) if x[0] is not None else float("-inf"),
                reverse=True,
            )
            if self.assume_normalized:
                self.ufun_min, self.ufun_max = 0.0, 1.0
            elif len(self.ordered_outcomes) < 1:
                self.ufun_max = self.ufun_min = self.utility_function.reserved_value
            else:
                if self.ufun_max is None:
                    self.ufun_max = self.ordered_outcomes[0][0]

                if self.ufun_min is None:
                    # we set the minimum utility to the minimum finite value above both reserved_value
                    for j in range(len(self.ordered_outcomes) - 1, -1, -1):
                        self.ufun_min = self.ordered_outcomes[j][0]
                        if self.ufun_min is not None and self.ufun_min > float("-inf"):
                            break
                    if (
                        self.ufun_min is not None
                        and self.ufun_min < self.reserved_value
                    ):
                        self.ufun_min = self.reserved_value
        else:
            if (
                self.ufun_min is None
                or self.ufun_max is None
                or self.best_outcome is None
                or self.worst_outcome is None
            ):
                mn, mx, self.worst_outcome, self.best_outcome = utility_range(
                    self.ufun, return_outcomes=True, issues=self._ami.issues
                )
                if self.ufun_min is None:
                    self.ufun_min = mn
                if self.ufun_max is None:
                    self.ufun_max = mx

        if self.ufun_min < self.reserved_value:
            self.ufun_min = self.reserved_value
        if self.ufun_max < self.ufun_min:
            self.ufun_max = self.ufun_min

        self.presorted = presort
        self.n_trials = 10

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if self.ufun_max is None or self.ufun_min is None:
            self.on_ufun_changed()
        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        u = self._utility_function(offer)
        if u is None or u < self.reserved_value:
            return ResponseType.REJECT_OFFER
        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if u >= asp and u > self.reserved_value:
            return ResponseType.ACCEPT_OFFER
        if asp < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.ufun_max is None or self.ufun_min is None:
            self.on_ufun_changed()
        if self.ufun_max < self.reserved_value:
            return None
        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if asp < self.reserved_value:
            return None
        if self.presorted:
            if len(self.ordered_outcomes) < 1:
                return None
            for i, (u, o) in enumerate(self.ordered_outcomes):
                if u is None:
                    continue
                if u < asp:
                    if u < self.reserved_value:
                        return None
                    if i == 0:
                        return self.ordered_outcomes[i][1]
                    if self.randomize_offer:
                        return random.sample(self.ordered_outcomes[:i], 1)[0][1]
                    return self.ordered_outcomes[i - 1][1]
            if self.randomize_offer:
                return random.sample(self.ordered_outcomes, 1)[0][1]
            return self.ordered_outcomes[-1][1]
        else:
            if asp >= 0.99999999999 and self.best_outcome is not None:
                return self.best_outcome
            if self.randomize_offer:
                return outcome_with_utility(
                    ufun=self._utility_function,
                    rng=(asp, float("inf")),
                    issues=self._ami.issues,
                )
            tol = self.tolerance
            for _ in range(self.n_trials):
                rng = self.ufun_max - self.ufun_min
                mx = min(asp + tol * rng, self.__last_offer_util)
                outcome = outcome_with_utility(
                    ufun=self._utility_function, rng=(asp, mx), issues=self._ami.issues,
                )
                if outcome is not None:
                    break
                tol = math.sqrt(tol)
            else:
                outcome = (
                    self.best_outcome
                    if self.__last_offer is None
                    else self.__last_offer
                )
            self.__last_offer_util = self.utility_function(outcome)
            self.__last_offer = outcome
            return outcome

