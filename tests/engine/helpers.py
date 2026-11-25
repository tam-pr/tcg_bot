"""Build exact game positions for rule and effect tests."""
from __future__ import annotations

from tcg_bot.engine import Action, legal_actions, new_game, step
from tcg_bot.engine.core import card_locations
from tcg_bot.engine.types import Slot, State

DECKS = ("lopunny", "mega_lucario")


def uids_named(state: State, p: int, name: str) -> list[int]:
    return sorted(u for u, r in state.catalog.refs.items() if r.owner == p and r.card.name == name)


def take(state: State, p: int, name: str, used: set) -> int:
    for u in uids_named(state, p, name):
        if u not in used:
            used.add(u)
            return u
    raise KeyError(f"player {p} has no more copies of {name!r}")


def scenario(decks=DECKS, seed=0, current=0, turn=3, first_player=0, turns_taken=(2, 1), hands=((), ()),
             actives=(None, None), benches=((), ()), prizes=(3, 3), discards=((), ()), decks_top=((), ()),
             stadium=None, energy=None, tools=None, damage=None, conditions=None) -> State:
    """A main-phase position. Names refer to cards of that player's deck. Every card not placed goes to the deck.
    energy / tools / damage / conditions: dicts keyed by (player, slot index) where slot 0 is the Active and
    1..5 the Bench, e.g. energy={(0, 0): ["Basic {F} Energy"]}."""
    s = new_game(decks, seed)
    used: set = set()
    for p in (0, 1):
        pl = s.players[p]
        pl.hand, pl.discard, pl.prizes, pl.bench, pl.active = [], [], [], [], None
        pl.hand = [take(s, p, n, used) for n in hands[p]]
        pl.discard = [take(s, p, n, used) for n in discards[p]]
        if actives[p]:
            u = take(s, p, actives[p], used)
            pl.active = Slot(pid=u, cards=[u], entered_turn=0)
        for n in benches[p]:
            u = take(s, p, n, used)
            pl.bench.append(Slot(pid=u, cards=[u], entered_turn=0))
        pl.setup_done = True
        pl.turns_taken = turns_taken[p]
    if stadium:
        p, n = stadium
        s.stadium = take(s, p, n, used)
    for (p, i), names in (energy or {}).items():
        slot = s.players[p].active if i == 0 else s.players[p].bench[i - 1]
        slot.energy = [take(s, p, n, used) for n in names]
    for (p, i), n in (tools or {}).items():
        slot = s.players[p].active if i == 0 else s.players[p].bench[i - 1]
        slot.tool = take(s, p, n, used)
    for (p, i), d in (damage or {}).items():
        slot = s.players[p].active if i == 0 else s.players[p].bench[i - 1]
        slot.damage = d
    for (p, i), cs in (conditions or {}).items():
        slot = s.players[p].active if i == 0 else s.players[p].bench[i - 1]
        slot.conditions = set(cs)
    for p in (0, 1):
        pl = s.players[p]
        top = [take(s, p, n, used) for n in decks_top[p]]
        rest = [u for u, r in s.catalog.refs.items() if r.owner == p and u not in used]
        rest.sort()
        pl.prizes = [rest.pop() for _ in range(prizes[p])]
        pl.deck = rest + list(reversed(top))          # decks_top[p][0] ends up on top
    s.phase, s.turn, s.current, s.first_player = "main", turn, current, first_player
    s.flags, s.frames, s.pending, s.lasting = {}, [], None, []
    assert len(card_locations(s)) == 120
    return s


def slot_of(state: State, p: int, i: int) -> Slot:
    return state.players[p].active if i == 0 else state.players[p].bench[i - 1]


def find(state: State, **kw) -> Action:
    """The single legal action matching the given fields (choice may be a predicate)."""
    out = []
    for a in legal_actions(state):
        ok = True
        for k, v in kw.items():
            av = getattr(a, k)
            if callable(v):
                ok = ok and bool(v(av))
            else:
                ok = ok and av == v
        if ok:
            out.append(a)
    assert len(out) >= 1, f"no legal action matches {kw}; legal: {legal_actions(state)}"
    return out[0]


def has(state: State, **kw) -> bool:
    try:
        find(state, **kw)
        return True
    except AssertionError:
        return False


def do(state: State, **kw) -> State:
    return step(state, find(state, **kw))


def choose(state: State, choice) -> State:
    return step(state, Action("choose", state.pending.player, choice=choice))


def name(state: State, uid: int) -> str:
    return state.catalog.refs[uid].card.name


def names(state: State, uids) -> list[str]:
    return sorted(name(state, u) for u in uids)


def move_id(state: State, card_name: str, move_name: str) -> str:
    for r in state.catalog.refs.values():
        if r.card.name == card_name:
            for a in r.card.attacks:
                if a.name == move_name:
                    return a.move_id
            for m in r.card.ability_ids + r.card.effect_ids:
                if state.catalog.effects[m].notes.split(",")[-1].strip().startswith(move_name) or move_name in state.catalog.effects[m].notes:
                    return m
    raise KeyError((card_name, move_name))
