"""Game engine. Public API:

    new_game(deck_ids, seed) -> State
    legal_actions(state) -> list[Action]      # empty only when the game is over
    step(state, action) -> State              # returns a new State; the input is unchanged
    observe(state, player) -> Observation     # only what that player may see
    winner(state) -> int | None

See reports/engine_interface.md for the contract.
"""
from .core import IllegalActionError, acting_player, card_locations, legal_actions, new_game, observe, step, step_inplace, winner
from .types import (Action, CardRef, CardStatic, Catalog, Observation, Pending, PlayerState, RuleSettings, Slot,
                    SlotView, State)

__all__ = ["new_game", "legal_actions", "step", "step_inplace", "observe", "winner", "acting_player", "card_locations",
           "IllegalActionError", "Action", "State", "Observation", "RuleSettings", "Slot", "SlotView",
           "PlayerState", "Pending", "Catalog", "CardRef", "CardStatic"]
