"""T12: encoding, masking, the agent, and a short training run that reproduces exactly."""
import random

import numpy as np
import pytest
import torch

from tcg_bot.engine import acting_player, legal_actions, new_game, observe, step_inplace
from tcg_bot.evaluate import play_game
from tcg_bot.agents import RandomAgent
from tcg_bot.model import Encoder, EncoderConfig, NeuralAgent, PolicyValueNet, batch
from tcg_bot.model.ppo import TrainConfig, train


def positions(n=120, seed=0):
    s = new_game(("lopunny", "mega_lucario"), seed)
    rng = random.Random(seed)
    for _ in range(n):
        legal = legal_actions(s)
        if not legal:
            return
        yield s, legal
        step_inplace(s, rng.choice(legal), validate=False)


@pytest.mark.parametrize("repr_", ["pca_bypass", "pca", "raw", "learned"])
def test_encoding_shapes_and_values(repr_):
    enc = Encoder(EncoderConfig(repr_, repr_ != "pca"))
    for s, legal in positions():
        o = observe(s, acting_player(s))
        x, slots, hand = enc.state(o)
        ax, ids = enc.actions(o, legal)
        assert x.shape == (enc.state_dim,) and ax.shape == (len(legal), enc.action_dim)
        assert np.isfinite(x).all() and np.isfinite(ax).all()
        assert slots.shape == (12,) and hand.shape == (20,) and ids.shape == (len(legal), 2)


def test_padding_is_masked_and_never_chosen():
    enc = Encoder()
    torch.manual_seed(0)
    net = PolicyValueNet(enc.state_dim, enc.action_dim, enc.n_cards)
    ag = NeuralAgent(net, enc, greedy=False)
    items = []
    for s, legal in positions(40):
        items.append(ag.encode(observe(s, acting_player(s)), legal))
    logits, value = net(*batch(items))
    amask = batch(items)[5]
    assert torch.isinf(logits[~amask]).all() and torch.isfinite(logits[amask]).all()
    assert (value.abs() <= 1).all()


def test_neural_agent_plays_legal_moves_from_observations_only():
    enc = Encoder()
    net = PolicyValueNet(enc.state_dim, enc.action_dim, enc.n_cards)
    rec = play_game((NeuralAgent(net, enc, greedy=False), RandomAgent(1)), ("lopunny", "mega_lucario"), 5)
    assert rec.result is not None


def test_short_training_run_is_reproducible():
    cfg = TrainConfig(iterations=2, games_per_iter=4, stage1_iters=1, snapshot_every=1, eval_games=0, seed=3)
    a = train(cfg, None, log=lambda r: None)
    b = train(cfg, None, log=lambda r: None)
    strip = lambda h: [{k: v for k, v in r.items() if k != "seconds"} for r in h]
    assert strip(a["history"]) == strip(b["history"])
    for p, q in zip(a["net"].parameters(), b["net"].parameters()):
        assert torch.equal(p, q)
