"""Playing whole games with the network, the stand-in and a random player, many games at a time."""
import random
from dataclasses import dataclass, field

import torch

from tcg_bot.engine.game import legal_actions, new_game, step
from tcg_bot.engine.observe import observe
from tcg_bot.engine.types import Action, Phase, State
from tcg_bot.model.encode import Encoded, encode
from tcg_bot.model.net import Net, collate
from tcg_bot.narrate import demo_choice

MATCHUPS = (("lopunny", "mega_lucario"), ("mega_lucario", "lopunny"))
SHAPING = 0.05                      # reward for each Prize card of lead gained, on top of +1 win / -1 loss


@torch.no_grad()
def decide(net: Net, batch: list[Encoded], greedy: bool) -> tuple[list[int], list[float], list[float]]:
    """For each decision: the chosen row, its log-probability and the value estimate."""
    logits, value = net(collate(batch))
    logp = torch.log_softmax(logits, dim=-1)
    chosen = logits.argmax(dim=-1) if greedy else torch.multinomial(logp.exp(), 1).squeeze(-1)
    return chosen.tolist(), logp.gather(1, chosen[:, None]).squeeze(1).tolist(), value.tolist()


def model_choice(net: Net, state: State, greedy: bool = True) -> Action:
    """One move by the model, for playing against a person. It is given the view only, never the state."""
    actions = legal_actions(state)
    view = observe(state, actions[0].player)
    encoded = encode(view, actions)
    return encoded.actions[decide(net, [encoded], greedy)[0][0]]


@dataclass
class Step:
    encoded: Encoded
    row: int
    logp: float
    value: float
    lead: int                       # Prize lead when this decision was made
    reward: float = 0.0


@dataclass
class Game:
    seed: int
    decks: tuple[str, str]
    seats: tuple[str, str]          # "learner", "past", "stand_in" or "random"
    state: State = field(init=False)
    pick: random.Random = field(init=False)
    steps: tuple[list[Step], list[Step]] = field(default_factory=lambda: ([], []))

    def __post_init__(self):
        self.state = new_game(self.decks, self.seed)
        self.pick = random.Random(self.seed)


def prize_lead(state: State, seat: int) -> int:
    if state.phase == Phase.SETUP:
        return 0
    return len(state.players[1 - seat].prizes) - len(state.players[seat].prizes)


def _finish(game: Game) -> None:
    """Rewards: the change in Prize lead between a player's decisions, and the result at the end."""
    for seat, steps in enumerate(game.steps):
        if not steps:
            continue                      # this seat did not learn
        leads = [s.lead for s in steps[1:]] + [prize_lead(game.state, seat)]
        for s, after in zip(steps, leads, strict=True):
            s.reward = SHAPING * (after - s.lead)
        if steps and game.state.winner is not None:
            steps[-1].reward += 1.0 if game.state.winner == seat else -1.0


def play(games: list[Game], net: Net, past: Net | None = None, greedy: bool = False) -> list[Game]:
    """Play every game to the end. Decisions that need a network are batched across games."""
    running = list(games)
    while running:
        waiting: dict[str, list[tuple[Game, int, Encoded]]] = {"learner": [], "past": []}
        for game in running:
            while game.state.phase != Phase.OVER:
                actions = legal_actions(game.state)
                seat = actions[0].player
                kind = game.seats[seat]
                if kind == "stand_in":
                    game.state = step(game.state, demo_choice(game.state, game.pick))
                elif kind == "random":
                    game.state = step(game.state, game.pick.choice(actions))
                else:
                    encoded = encode(observe(game.state, seat), actions)
                    if len(encoded.actions) == 1:                 # nothing to decide
                        game.state = step(game.state, encoded.actions[0])
                        continue
                    waiting[kind].append((game, seat, encoded))
                    break
        for kind, batch in waiting.items():
            if not batch:
                continue
            rows, logps, values = decide(past if kind == "past" else net, [b[2] for b in batch], greedy)
            for (game, seat, encoded), row, logp, value in zip(batch, rows, logps, values, strict=True):
                if kind == "learner":
                    game.steps[seat].append(Step(encoded, row, logp, value, prize_lead(game.state, seat)))
                game.state = step(game.state, encoded.actions[row])
        running = [g for g in running if g.state.phase != Phase.OVER]
    for game in games:
        _finish(game)
    return games
