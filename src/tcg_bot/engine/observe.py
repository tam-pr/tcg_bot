"""What one player may see. A player (rule-based, search or model) is given this and never the State.

Hidden: the opponent's hand (except cards it has had to show), both decks, all Prize cards, and during setup
the opponent's face-down Pokémon. An Observation holds copies, so changing it changes nothing in the game.

Not recorded yet: the hands an opponent shows when taking a mulligan.
"""
from dataclasses import dataclass, replace

from tcg_bot.engine.types import LastingEffect, PendingChoice, Phase, PokemonInPlay, State, TurnFlags, owner_of

CARD_CHOICES = {"pick", "attach_pick", "energy_to_hand", "retreat_cost"}   # their options are card uids


@dataclass(frozen=True)
class PlayerView:
    active: PokemonInPlay | None
    bench: tuple[PokemonInPlay, ...]
    face_down: int                       # Pokémon in play that cannot be identified yet (setup)
    hand: tuple[int, ...]                # only the cards the viewer knows
    hand_count: int
    deck_count: int
    prize_count: int
    discard: tuple[int, ...]
    playing: tuple[int, ...]             # the Trainer card being resolved
    flags: TurnFlags
    mulligans: int
    knocked_out_on_turn: int | None


@dataclass(frozen=True)
class Observation:
    player: int
    phase: Phase
    turn: int
    current: int
    first_player: int
    waiting_for: int | None              # who has to act now; None when the game is over
    me: PlayerView
    opponent: PlayerView
    stadium: int | None
    lasting: tuple[LastingEffect, ...]
    pending: PendingChoice | None        # only when the choice is the viewer's
    winner: int | None
    end_reason: str | None
    card_ids: dict[int, int]             # uid -> card_id for the cards the viewer can identify, and no others
    my_deck_list: tuple[int, ...]        # the viewer knows which 60 cards it brought, not their order


def _view(state: State, player: int, viewer: int) -> PlayerView:
    p = state.players[player]
    mine = player == viewer
    face_down = not mine and state.phase == Phase.SETUP
    known = p.hand if mine else [u for u in p.hand if u in p.revealed]
    return PlayerView(
        active=None if face_down or p.active is None else p.active.clone(),
        bench=() if face_down else tuple(m.clone() for m in p.bench),
        face_down=len(p.pokemon()) if face_down else 0,
        hand=tuple(sorted(known)),       # sorted, so the order says nothing
        hand_count=len(p.hand), deck_count=len(p.deck), prize_count=len(p.prizes),
        discard=tuple(p.discard), playing=tuple(p.playing), flags=p.flags.clone(),
        mulligans=p.mulligans, knocked_out_on_turn=p.knocked_out_on_turn)


def observe(state: State, player: int) -> Observation:
    me, opponent = _view(state, player, player), _view(state, 1 - player, player)
    pending = state.pending if state.pending and state.pending.player == player else None
    if pending and pending.kind in CARD_CHOICES:                    # sorted: not the order they lie in the deck
        pending = replace(pending, options=tuple(sorted(pending.options)))

    seen: list[int] = [] if state.stadium is None else [state.stadium]
    for view in (me, opponent):
        seen += view.hand + view.discard + view.playing
        for mon in ([view.active] if view.active else []) + list(view.bench):
            seen += mon.stack + mon.energy + ([] if mon.tool is None else [mon.tool])
    if pending and pending.kind in CARD_CHOICES:                    # searching a deck shows what is found
        seen += pending.options + pending.picked
    if pending and pending.kind == "attach_target":
        seen.append(pending.subject)

    if state.phase == Phase.OVER:
        waiting = None
    else:
        waiting = state.pending.player if state.pending else state.current
    return Observation(
        player=player, phase=state.phase, turn=state.turn, current=state.current, first_player=state.first_player,
        waiting_for=waiting, me=me, opponent=opponent, stadium=state.stadium, lasting=tuple(state.lasting),
        pending=pending, winner=state.winner, end_reason=state.end_reason,
        card_ids={u: state.card_ids[u] for u in seen},
        my_deck_list=tuple(sorted(c for u, c in enumerate(state.card_ids) if owner_of(u) == player)))
