"""The game's public functions: new_game, legal_actions and step.

So far: the opening (coin flip, hands, mulligans, placing Pokémon, Prizes) and the draw that starts a turn.
Main-phase actions are not written yet and raise NotImplementedError.

Order follows the `plays` table: hands, mulligans (with the bonus draw), Active and Bench, then Prizes.
`step` never changes the state it is given; it returns a new one.
"""
from tcg_bot.engine.catalog import Card, catalog
from tcg_bot.engine.types import PLAYERS, Action, ActionKind, PendingChoice, Phase, PokemonInPlay, State, TurnFlags


def card_of(state: State, uid: int) -> Card:
    return catalog().cards[state.card_ids[uid]]


def new_game(deck_ids: tuple[str, str], seed: int) -> State:
    """A state waiting for the winner of the coin flip to say whether they go first."""
    decks = catalog().decks
    unknown = [d for d in deck_ids if d not in decks]
    if unknown:
        raise ValueError(f"unknown deck_id {unknown}")
    state = State.initial((list(decks[deck_ids[0]]), list(decks[deck_ids[1]])), seed)
    winner = state.rng.randrange(2)
    state.pending = PendingChoice(player=winner, kind="go_first", options=(0, 1))     # 1 = "I go first"
    return state


def legal_actions(state: State) -> list[Action]:
    """Everything the player to move may do. Empty only when the game is over."""
    if state.phase == Phase.OVER:
        return []
    if state.pending:
        return [Action(ActionKind.CHOOSE, state.pending.player, option=o) for o in state.pending.options]
    if state.phase == Phase.SETUP:
        return _placement_actions(state)
    raise NotImplementedError(f"{state.phase} phase is not written yet")


def step(state: State, action: Action) -> State:
    if action not in legal_actions(state):
        raise ValueError(f"illegal action: {action}")
    new = state.clone()
    if action.kind == ActionKind.CHOOSE:
        _answer(new, action.option)
    else:
        _place(new, action)
    return new


# ---------------------------------------------------------------- drawing
def draw(state: State, player: int, count: int = 1) -> int:
    """Move up to `count` cards from the top of the deck to the hand. Returns how many moved."""
    p = state.players[player]
    n = min(count, len(p.deck))
    p.hand += p.deck[:n]
    del p.deck[:n]
    return n


def start_turn(state: State, player: int) -> None:
    state.turn += 1
    state.current = player
    state.players[player].flags = TurnFlags()
    for p in state.players:
        for mon in p.pokemon():
            mon.abilities_used.clear()
    if draw(state, player) == 0:                      # cannot draw at the start of the turn
        state.phase, state.winner, state.end_reason = Phase.OVER, 1 - player, "deck_out"


# ---------------------------------------------------------------- the opening
def _has_basic(state: State, player: int) -> bool:
    return any(card_of(state, u).is_basic_pokemon for u in state.players[player].hand)


def _deal_opening_hand(state: State, player: int) -> None:
    """Shuffle and draw until the hand has a Basic Pokémon, counting the mulligans."""
    p = state.players[player]
    if not any(card_of(state, u).is_basic_pokemon for u in p.deck + p.hand):
        raise ValueError(f"player {player}: the deck has no Basic Pokémon")
    while True:
        state.rng.shuffle(p.deck)
        draw(state, player, catalog().constants["opening_hand_size"])
        if _has_basic(state, player):
            return
        p.mulligans += 1
        p.deck += p.hand
        p.hand.clear()


def _answer(state: State, option: int) -> None:
    pending, state.pending = state.pending, None
    if pending.kind == "go_first":
        state.first_player = pending.player if option == 1 else 1 - pending.player
        state.current = state.first_player
        for player in PLAYERS:
            _deal_opening_hand(state, player)
        extra = state.players[0].mulligans - state.players[1].mulligans
        if extra:                                     # the player with fewer mulligans may draw the difference
            chooser = 1 if extra > 0 else 0
            state.pending = PendingChoice(player=chooser, kind="mulligan_draw",
                                          options=tuple(range(abs(extra) + 1)))
    elif pending.kind == "mulligan_draw":
        draw(state, pending.player, option)
    else:
        raise NotImplementedError(f"choice {pending.kind!r} is not written yet")


def _placement_actions(state: State) -> list[Action]:
    player = state.current
    p = state.players[player]
    basics = [u for u in p.hand if card_of(state, u).is_basic_pokemon]
    if p.active is None:
        return [Action(ActionKind.PLACE_ACTIVE, player, card=u) for u in basics]
    out = [Action(ActionKind.SETUP_DONE, player)]
    if len(p.bench) < catalog().constants["bench_max"]:
        out += [Action(ActionKind.PLACE_BENCH, player, card=u) for u in basics]
    return out


def _place(state: State, action: Action) -> None:
    p = state.players[action.player]
    if action.kind == ActionKind.SETUP_DONE:
        if action.player == state.first_player:
            state.current = 1 - state.first_player
        else:
            _finish_setup(state)
        return
    p.hand.remove(action.card)
    mon = PokemonInPlay(stack=[action.card], in_play_since=0)
    if action.kind == ActionKind.PLACE_ACTIVE:
        p.active = mon
    else:
        p.bench.append(mon)


def _finish_setup(state: State) -> None:
    for p in state.players:
        p.prizes = p.deck[:state.prize_count]
        del p.deck[:state.prize_count]
    state.phase = Phase.MAIN
    start_turn(state, state.first_player)


# ---------------------------------------------------------------- try it: uv run python -m tcg_bot.engine.game
def _demo(seed: int = 0) -> None:
    state = new_game(("lopunny", "mega_lucario"), seed)
    while state.phase == Phase.SETUP:
        state = step(state, legal_actions(state)[-1])
    names = ("lopunny", "mega_lucario")
    print(f"{names[state.first_player]} goes first")
    for i, p in enumerate(state.players):
        print(f"\n{names[i]} (mulligans: {p.mulligans})")
        print("  active:", card_of(state, p.active.card).name)
        print("  bench: ", [card_of(state, m.card).name for m in p.bench])
        print("  hand:  ", [card_of(state, u).name for u in p.hand])
        print(f"  deck {len(p.deck)} cards, prizes {len(p.prizes)}")


if __name__ == "__main__":
    _demo()