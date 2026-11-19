# Pokémon TCG deck-simulation pipeline

## Goal

Give the system a 60-card deck. It plays simulated games against an arbitrary opponent deck, tracks what it can't see (its own Prizes, the opponent's deck), and picks plays with a locally trained model. No LLMs or external AI services.

## The pipeline

### 1. Clean the data

Parse `EN_Card_Data_R2_full.csv`: strip `n/a` values and stray whitespace, turn `{F}{F}●` into energy counts, and split `30×` and `30+` into a number plus a modifier.

*Why:* every later step depends on it. The CSV has one row per move, so cards repeat.

### 2. Build three tables

- `cards`: stats and rule flags.
- `moves`: every attack, ability and Trainer/Energy effect, with cost, damage, timing, requirements, targets, tags, extracted parameters and a parse status.
- `plays`: the generic turn actions (attach Energy, evolve, retreat, attack…) and their global rules, hand-written once.

*Why:* the model needs structured data, not prose. Global rules (one Supporter per turn) are kept separate from card-specific requirements (in the effect text).

### 3. Tag the effects

Binary multi-label flags like `heal`, `status`, `damage_bench_opp`, `switch_opp_active`. Regex first (a quick test with 18 tags tagged 79% of the Pokémon effect texts). Then hand-label a small set and train your own classifier (TF-IDF plus logistic regression) for the rest. Anything unrecognized gets an `unknown` flag.

*Why:* masking numbers alone left too many one-off templates (587 of 757 appeared once), so the long tail needs a real classifier. Tags become model features.

### 4. Build the card feature matrix and apply PCA

Combine the tags, stats (HP, retreat, cost counts, damage) and type one-hots. Standardize, then run PCA. Check the explained-variance curve to choose the number of components, and inspect the loadings. BERTopic or KMeans on the components is an optional way to explore card groups.

*Why:* it gives each card a compact vector for the model, and a plot of how cards group. Fit it on training cards only if you hold any out.

### 5. Add decks and a validator

Load decklists and check the construction rules (60 cards, at least 1 Basic Pokémon, max 4 copies by name with Basic Energy unlimited, 1 ACE SPEC). Verify the full rule list against the official rulebook before coding it. Build a `decks` table of known archetypes.

*Why:* the validator catches bad input early. The archetype list is what makes opponent inference possible.

### 6. Build belief trackers

- **Prizes:** unseen pool = deck list − cards seen. Those cards are split between the deck and the 6 Prizes, so prize odds come from counting.
- **Opponent deck:** a Bayesian update over the known archetypes as the opponent's cards are revealed.

*Why:* this is arithmetic, not learning, so it is more reliable than hoping a network figures it out. The probabilities become model inputs.

### 7. Build the game engine

Setup, turn structure, attacks, status conditions, Knock Outs, Prizes and win conditions, for just the cards in the chosen decks. It exposes `legal_actions(state)` and `step(state, action)`.

*Why:* this is the largest piece and the biggest risk. A model trained on a buggy engine learns the wrong game, so it needs tests.

### 8. Build baseline agents

Random, then a hand-written heuristic. Optional: determinized MCTS using the belief tracker.

*Why:* they validate the engine, and the learned model needs something to be measured against.

### 9. Train the model

A PyTorch network trained by self-play (PPO with action masking). Cards enter as the PCA vectors, the state encoder takes the board plus the belief vectors, the policy scores each legal action, and a value head predicts win probability.

*Why:* this is the genuine ML part. Encoding cards by their effects lets it generalize beyond card IDs.

### 10. Evaluate

Thousands of games per matchup: win rate against random and heuristic players and against held-out decks, plus diagnostics such as the rate of opening without a Basic.

*Why:* it shows whether the agent really learned, and whether it overfits the known decks.

## Scope and timeline

Support **2 or 3 decks (about 30 to 60 cards)**, not the whole pool.

| Week | Steps | Result |
|---|---|---|
| 1 | 1–6 | Data tables, tags, PCA, deck validator, belief trackers |
| 2 | 7–8 | Engine and baseline agents |
| 3 | 9–10 | Trained model and evaluation |

Fallback if training doesn't pay off in time: a smaller learned component (for example choosing among a few options) on top of the heuristic player.

## What the data looks like (from `EN_Card_Data_R2_full.csv`)

- 2,273 rows, one per move.
- 2,030 rows are Pokémon moves from 1,196 distinct cards. The rest are Energy and Trainer cards.
- 1,436 Pokémon moves have effect text, and 915 of those texts are unique. About 594 moves are plain damage with no text.
- Median effect length is about 93 characters.
- Abilities and attacks share the `Effect Explanation` column. Abilities are only identifiable by phrases like "Once during your turn, you may use this Ability".

## Main risks

1. **Engine bugs.** Budget testing time.
2. **Training time.** Self-play needs many games, and a laptop is a real limit.
3. **Scope creep.** Adding one more deck or card family usually costs more than it seems.

## Open questions

- Where the decklists come from.
- Hours per week and what must exist on the last day.
- Whether the PCA requirement names a specific data type.
- Whether to include the optional MCTS player.
