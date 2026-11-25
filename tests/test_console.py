"""T9: typed commands map to legal actions, bad commands explain themselves, a scripted full game passes."""
import random

import pytest

from tcg_bot.agents import HeuristicAgent, RandomAgent
from tcg_bot.console import CommandError, Console, candidates, describe, parse, render
from tcg_bot.engine import legal_actions

from tests.engine.helpers import scenario


def test_every_legal_action_round_trips_through_its_command():
    for seed in range(15):
        c = Console(seed=seed, opponent=RandomAgent(seed))
        c.start()
        rng = random.Random(seed)
        while not c.over():
            for a in legal_actions(c.state):
                b = parse(c.state, describe(c.state, a))
                assert describe(c.state, b) == describe(c.state, a)
            c.handle(describe(c.state, rng.choice(legal_actions(c.state))))


def test_examples_from_the_task():
    s = scenario(decks=("mega_lucario", "lopunny"), hands=(("Lunatone", "Basic {F} Energy"), ()),
                 actives=("Solrock", "Snorunt"), benches=(("Riolu",), ()))
    assert parse(s, "lunatone to bench").kind == "bench_basic"
    a = parse(s, "basic energy to solrock")
    assert a.kind == "attach_energy" and s.catalog.card(a.source).name == "Basic {F} Energy"
    s.flags["energy_attached"] = True
    with pytest.raises(CommandError, match="already attached"):
        parse(s, "basic energy to solrock")


def test_ambiguous_and_unknown_commands_list_options():
    s = scenario(decks=("mega_lucario", "lopunny"), hands=(("Basic {F} Energy",), ()), actives=("Solrock", "Snorunt"),
                 benches=(("Riolu",), ()))
    with pytest.raises(CommandError, match="ambiguous"):
        parse(s, "basic energy")
    with pytest.raises(CommandError, match="unknown word"):
        parse(s, "pikachu to bench")


def test_scripted_full_game_against_the_heuristic():
    c = Console(decks=("mega_lucario", "lopunny"), seed=3, opponent=HeuristicAgent(3))
    first = c.start()
    assert first == "" or first.startswith("model: ")
    script = HeuristicAgent(99)
    from tcg_bot.engine import observe
    lines = 0
    while not c.over():
        a = script.act(observe(c.state, 0), legal_actions(c.state))
        reply = c.handle(describe(c.state, a))
        assert not reply.startswith(("ambiguous", "unknown", "that is not")), reply
        lines += 1
    assert "game over" in c.handle("end turn") and lines > 10
    assert "you" in render(c.state, 0)
