import random

import pytest
import torch

from tcg_bot.engine.game import legal_actions, new_game, step
from tcg_bot.engine.observe import observe
from tcg_bot.engine.types import Phase
from tcg_bot.model import net as network
from tcg_bot.model.encode import encode
from tcg_bot.model.evaluate import EVAL_SEED, win_rate
from tcg_bot.model.net import Net, collate, load, save
from tcg_bot.model.player import Game, Step, model_choice, play
from tcg_bot.model.train import advantages, new_games, train


def decisions(seed=0, n=40):
    pick, state, out = random.Random(seed), new_game(("lopunny", "mega_lucario"), seed), []
    while state.phase != Phase.OVER and len(out) < n:
        actions = legal_actions(state)
        out.append(encode(observe(state, actions[0].player), actions))
        state = step(state, pick.choice(actions))
    return out


def test_one_score_per_move_and_padding_is_never_chosen():
    batch = decisions()
    logits, value = Net()(collate(batch))
    probabilities = torch.softmax(logits, dim=-1)
    assert logits.shape == (len(batch), max(len(e.moves) for e in batch)) and value.shape == (len(batch),)
    for e, row in zip(batch, probabilities, strict=True):
        assert row[len(e.moves):].sum() == 0 and torch.isclose(row.sum(), torch.tensor(1.0))
    assert value.abs().max() <= 1


def test_a_decision_scores_the_same_alone_or_in_a_batch():
    torch.manual_seed(0)
    net, batch = Net().eval(), decisions()
    together, _ = net(collate(batch))
    alone, _ = net(collate(batch[3:4]))
    assert torch.allclose(together[3, :alone.shape[1]], alone[0], atol=1e-5)


def test_the_model_only_plays_legal_moves_and_games_finish():
    torch.manual_seed(0)
    net = Net().eval()
    state = new_game(("lopunny", "mega_lucario"), 3)
    while state.phase != Phase.OVER:
        state = step(state, model_choice(net, state, greedy=False))          # step raises on an illegal move
    assert state.end_reason


def test_rewards_add_up_to_the_result():
    torch.manual_seed(0)
    for game in play([Game(seed=s, decks=("lopunny", "mega_lucario"), seats=("learner", "learner")) for s in range(4)],
                     Net().eval()):
        for seat, steps in enumerate(game.steps):
            result = 0 if game.state.winner is None else (1 if game.state.winner == seat else -1)
            lead = len(game.state.players[1 - seat].prizes) - len(game.state.players[seat].prizes)
            assert sum(s.reward for s in steps) == pytest.approx(result + 0.05 * (lead - steps[0].lead))


def test_advantages_on_a_two_step_game():
    steps = [Step(None, 0, 0.0, value=0.5, lead=0, reward=0.0), Step(None, 0, 0.0, value=0.2, lead=0, reward=1.0)]
    (a0, r0), (a1, r1) = advantages(steps, gamma=1.0, lam=1.0)
    assert (a1, r1) == (pytest.approx(0.8), pytest.approx(1.0)) and (a0, r0) == (pytest.approx(0.5), pytest.approx(1.0))


def test_training_never_uses_the_evaluation_seeds():
    config = {"games": 64, "stand_in": 0.35, "past": 0.15, "seed": 0}
    seeds = [g.seed for g in new_games(config, 5000, random.Random(0), True)]
    assert max(seeds) < EVAL_SEED and len(set(seeds)) == len(seeds)


def test_a_saved_model_loads_and_answers_the_same(tmp_path):
    torch.manual_seed(1)
    net, batch = Net().eval(), collate(decisions())
    save(net, tmp_path / "model.pt")
    assert torch.allclose(load(tmp_path)(batch)[0], net(batch)[0])


def test_a_model_for_different_inputs_is_refused(tmp_path, monkeypatch):
    save(Net(), tmp_path / "model.pt")
    monkeypatch.setattr(network, "sizes", lambda: {"cards": 1})
    with pytest.raises(ValueError):
        load(tmp_path)


def test_a_short_training_run_writes_a_model_and_changes_it(tmp_path):
    folder = train("t", 2, folder=tmp_path / "t", say=lambda line: None, games=4, minibatch=64, epochs=1,
                   snapshot_every=1, eval_every=2, eval_games=4)
    lines = (folder / "metrics.csv").read_text().strip().splitlines()
    assert len(lines) == 3 and (folder / "model.pt").exists()
    torch.manual_seed(0)
    assert not all(torch.equal(a, b) for a, b in zip(Net().parameters(), load(folder).parameters(), strict=True))
    with pytest.raises(SystemExit):
        train("t", 1, folder=folder)                                         # will not overwrite by accident


def test_win_rate_is_a_share_of_the_games():
    torch.manual_seed(0)
    result = win_rate(Net().eval(), "random", 8)
    assert 0 <= result["win_rate"] <= 1 and result["games"] == 8 and set(result["by_deck"]) == {"lopunny", "mega_lucario"}
