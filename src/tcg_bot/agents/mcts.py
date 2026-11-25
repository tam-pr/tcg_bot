"""Determinized Monte Carlo tree search (task T11, optional).

The agent never reads the true hidden state. For each decision it:
1. Samples a full game state consistent with its Observation (determinize): the opponent's deck list
   from OpponentDeckBelief, then a random split of the unseen cards into deck, Prizes and the
   opponent's hand. A random split of my own unseen cards between deck and Prizes is exactly the
   hypergeometric PrizeBelief.
2. Runs UCT over its own actions until the end of its turn, with heuristic play-outs that are cut after
   `rollout_actions` actions and scored by a simple evaluation (Prize cards left, HP left).
3. Adds up the root visit counts over all samples and plays the most visited action.

Choices inside effects (pending choices) and setup are delegated to the heuristic, because the
Observation does not carry the interpreter's position (state.frames) needed to rebuild them.
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter

from tcg_bot.beliefs import OpponentDeckBelief, archetype_lists
from tcg_bot.engine import acting_player, legal_actions, observe, step_inplace
from tcg_bot.engine.catalog import make_catalog
from tcg_bot.engine.types import (UID_STRIDE, Action, Observation, PlayerState, RuleSettings, Slot, SlotView, State)

from . import Agent
from .heuristic import HeuristicAgent


def _my_deck_id(obs: Observation) -> str:
    for d in archetype_lists():
        if make_catalog(d, d).deck_lists[0] == obs.my_deck_list_card_ids:
            return d
    raise ValueError("my deck list does not match a known deck")


def _slot(v: SlotView, remap) -> Slot:
    return Slot(pid=remap(v.pid), cards=[remap(u) for u in v.cards], energy=[remap(u) for u in v.energy],
                tool=remap(v.tool) if v.tool is not None else None, damage=v.damage, conditions=set(v.conditions),
                entered_turn=v.entered_turn, evolved_turn=v.evolved_turn, moved_to_active_turn=v.moved_to_active_turn,
                paralyzed_turn=v.paralyzed_turn, lockouts=dict(v.lockouts), used=dict(v.used))


def determinize(obs: Observation, belief: OpponentDeckBelief, rng: random.Random,
                settings: RuleSettings | None = None) -> State:
    """A full State consistent with what `obs.player` can see. Hidden zones are sampled."""
    me, op = obs.player, 1 - obs.player
    mine = _my_deck_id(obs)
    post = belief.update(obs)
    decks = list(post)
    opp_deck = rng.choices(decks, weights=[post[d] for d in decks])[0]
    pair = (mine, opp_deck) if me == 0 else (opp_deck, mine)
    cat = make_catalog(*pair)

    # opponent uids seen in this game may name different cards in the sampled list: remap by card_id
    opp_ids = {u: cat.refs[u].card.card_id for u in cat.refs if cat.refs[u].owner == op}
    seen_opp: dict[int, int] = {}
    for u, c in zip(obs.opp_discard, obs.opp_discard_card_ids):
        seen_opp[u] = c
    for v in (obs.opp_active, *obs.opp_bench):
        if v is not None:
            for u, c in zip(v.cards, v.card_ids):
                seen_opp[u] = c
            for u, c in zip(v.energy, v.energy_card_ids):
                seen_opp[u] = c
            if v.tool is not None:
                seen_opp[v.tool] = v.tool_card_id
    if obs.stadium is not None and obs.stadium // UID_STRIDE == op:
        seen_opp[obs.stadium] = obs.stadium_card_id
    mapping: dict[int, int] = {}
    free = {u for u in opp_ids if not (u in seen_opp and opp_ids[u] == seen_opp[u])}
    for u, c in seen_opp.items():
        if opp_ids.get(u) == c:
            mapping[u] = u
        else:
            cand = sorted(x for x in free if opp_ids[x] == c)
            if not cand:
                raise ValueError(f"card {c} is not in sampled deck {opp_deck}")
            mapping[u] = cand[0]
            free.discard(cand[0])
    remap = lambda u: mapping.get(u, u)

    players = [PlayerState(), PlayerState()]
    pm = players[me]
    pm.hand = list(obs.my_hand)
    pm.discard = list(obs.my_discard)
    pm.active = _slot(obs.my_active, lambda u: u) if obs.my_active else None
    pm.bench = [_slot(v, lambda u: u) for v in obs.my_bench]
    seen_me = set(pm.hand) | set(pm.discard) | {u for s in pm.in_play() for u in s.cards + s.energy + ([s.tool] if s.tool is not None else [])}
    if obs.stadium is not None and obs.stadium // UID_STRIDE == me:
        seen_me.add(obs.stadium)
    unseen_me = sorted(u for u in cat.refs if cat.refs[u].owner == me and u not in seen_me)
    rng.shuffle(unseen_me)
    pm.prizes = unseen_me[:obs.my_prize_count]
    pm.deck = unseen_me[obs.my_prize_count:]

    po = players[op]
    po.discard = [remap(u) for u in obs.opp_discard]
    po.active = _slot(obs.opp_active, remap) if obs.opp_active else None
    po.bench = [_slot(v, remap) for v in obs.opp_bench]
    seen_o = set(mapping.values())
    unseen_o = sorted(u for u in opp_ids if u not in seen_o)
    rng.shuffle(unseen_o)
    po.hand = unseen_o[:obs.opp_hand_count]
    po.prizes = unseen_o[obs.opp_hand_count:obs.opp_hand_count + obs.opp_prize_count]
    po.deck = unseen_o[obs.opp_hand_count + obs.opp_prize_count:]
    for p in (0, 1):
        players[p].turns_taken = obs.turns_taken[p]
        players[p].ko_turns = list(obs.ko_turns[p])
        players[p].setup_done = True
    flags = {k: (list(v) if isinstance(v, tuple) else v) for k, v in obs.flags}
    st = State(catalog=cat, settings=settings or RuleSettings(), players=players, rng=random.Random(rng.random()),
               phase=obs.phase, turn=obs.turn, current=obs.current, first_player=obs.first_player,
               stadium=remap(obs.stadium) if obs.stadium is not None else None, flags=flags,
               lasting=[dict(le) for le in obs.lasting])
    return st


class _Node:
    __slots__ = ("n", "w", "children", "untried")

    def __init__(self):
        self.n, self.w, self.children, self.untried = 0, 0.0, {}, None


def evaluate(state: State, me: int) -> float:
    """1 win, 0 loss, 0.5 draw; otherwise a score in (0, 1) from Prize cards left and HP left."""
    if state.phase == "over":
        return 0.5 if state.winner is None else float(state.winner == me)
    mine, theirs = state.players[me], state.players[1 - me]
    prize = len(theirs.prizes) - len(mine.prizes)

    def hp(pl):
        return sum(max(0, (state.catalog.card(s.top).hp or 0) - s.damage) for s in pl.in_play())
    return 1 / (1 + math.exp(-(0.6 * prize + 0.004 * (hp(mine) - hp(theirs)))))


class MCTSAgent(Agent):
    name = "mcts"

    def __init__(self, determinizations: int = 4, iterations: int = 24, rollout_actions: int = 40,
                 c: float = 1.0, seed: int = 0):
        self.D, self.iters, self.depth, self.c = determinizations, iterations, rollout_actions, c
        self.rng = random.Random(seed)
        self.fallback = HeuristicAgent(seed)
        self.rollout_policy = HeuristicAgent(seed + 7)
        self.decisions, self.seconds = 0, 0.0          # searched decisions and the time they took

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
            self.fallback.reset(seed)

    def act(self, obs: Observation, legal: list[Action]) -> Action:
        if obs.phase != "main" or legal[0].kind == "choose" or len(legal) == 1:
            return self.fallback.act(obs, legal)
        t0 = time.perf_counter()
        me = obs.player
        belief = OpponentDeckBelief()
        visits: Counter = Counter()
        for _ in range(self.D):
            root_state = determinize(obs, belief, self.rng)
            assert set(legal_actions(root_state)) == set(legal), "determinized state disagrees on legal actions"
            root = _Node()
            for _ in range(self.iters):
                self._iterate(root_state, root, me, obs.turn)
            for a, ch in root.children.items():
                visits[a] += ch.n
        self.decisions += 1
        self.seconds += time.perf_counter() - t0
        return max(legal, key=lambda a: (visits[a], -legal.index(a)))

    def _iterate(self, root_state: State, root: _Node, me: int, turn: int) -> None:
        s = root_state.copy()
        node, path = root, [root]
        while s.phase != "over" and acting_player(s) == me and s.turn == turn and s.pending is None:
            acts = legal_actions(s)
            if node.untried is None:
                node.untried = list(acts)
                self.rng.shuffle(node.untried)
            if node.untried:
                a = node.untried.pop()
                step_inplace(s, a, validate=False)
                child = node.children[a] = _Node()
                path.append(child)
                break
            logn = math.log(node.n + 1)
            a = max(node.children, key=lambda x: node.children[x].w / (node.children[x].n + 1e-9)
                    + self.c * math.sqrt(logn / (node.children[x].n + 1e-9)))
            step_inplace(s, a, validate=False)
            node = node.children[a]
            path.append(node)
        v = self._rollout(s, me)
        for nd in path:
            nd.n += 1
            nd.w += v

    def _rollout(self, s: State, me: int) -> float:
        for _ in range(self.depth):
            acts = legal_actions(s)
            if not acts:
                break
            p = acting_player(s)
            step_inplace(s, self.rollout_policy.act(observe(s, p), acts), validate=False)
        return evaluate(s, me)
