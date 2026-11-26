"""The network: reads an encoded view, scores every legal move and estimates how the game is going.

Nothing is pretrained. Shapes come from the encoder, so a different set of cards needs a new network.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tcg_bot.model.encode import GAME_FEATURES, KINDS, SLOT_FEATURES, SLOTS, Encoded, vocabulary

EMBED, SLOT_OUT, HIDDEN = 8, 48, 128


def sizes() -> dict[str, int]:
    vocab = vocabulary()
    vector = vocab.vectors.shape[1]
    return {"cards": len(vocab.card_ids), "slot": len(SLOT_FEATURES) + vector, "slots": len(SLOTS),
            "counts": 5 * len(vocab.card_ids), "game": len(GAME_FEATURES),
            "move": len(KINDS) + 1 + vector + 1 + len(SLOTS) + 3 + 2}


def collate(batch: list[Encoded]) -> dict[str, torch.Tensor]:
    """Stack decisions into tensors. Move lists differ in length, so they are padded and masked."""
    most = max(len(e.moves) for e in batch)
    moves = np.zeros((len(batch), most, batch[0].moves.shape[1]), dtype=np.float32)
    move_cards = np.zeros((len(batch), most), dtype=np.int64)
    mask = np.zeros((len(batch), most), dtype=bool)
    for i, e in enumerate(batch):
        n = len(e.moves)
        moves[i, :n], move_cards[i, :n], mask[i, :n] = e.moves, e.move_cards, True
    out = {name: torch.from_numpy(np.stack([getattr(e, name) for e in batch]))
           for name in ("slots", "slot_cards", "counts", "game", "context_cards")}
    out |= {"moves": torch.from_numpy(moves), "move_cards": torch.from_numpy(move_cards),
            "mask": torch.from_numpy(mask)}
    return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        s = sizes()
        self.card = nn.Embedding(s["cards"] + 1, EMBED, padding_idx=0)
        self.slot = nn.Sequential(nn.Linear(s["slot"] + EMBED, SLOT_OUT), nn.ReLU())
        flat = s["slots"] * SLOT_OUT + s["counts"] + s["game"] + 3 * EMBED
        self.body = nn.Sequential(nn.Linear(flat, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(HIDDEN + s["move"] + EMBED, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
        self.value = nn.Sequential(nn.Linear(HIDDEN, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        with torch.no_grad():                       # start close to choosing every move equally often
            self.policy[-1].weight.mul_(0.01)
            self.policy[-1].bias.zero_()

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (a score per move, with padding at -1e9) and (value from -1 losing to +1 winning)."""
        slots = self.slot(torch.cat([batch["slots"], self.card(batch["slot_cards"])], dim=-1))
        state = self.body(torch.cat([slots.flatten(1), batch["counts"].flatten(1), batch["game"],
                                     self.card(batch["context_cards"]).flatten(1)], dim=-1))
        moves = torch.cat([batch["moves"], self.card(batch["move_cards"])], dim=-1)
        spread = state.unsqueeze(1).expand(-1, moves.shape[1], -1)
        logits = self.policy(torch.cat([spread, moves], dim=-1)).squeeze(-1)
        return logits.masked_fill(~batch["mask"], -1e9), self.value(state).squeeze(-1)


# ---------------------------------------------------------------- saving
def save(net: Net, path: Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": net.state_dict(), "card_ids": list(vocabulary().card_ids), "sizes": sizes(), **extra}, path)


def load(path: Path) -> Net:
    """`path` is a model.pt file or the folder that holds it."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["card_ids"] != list(vocabulary().card_ids) or saved["sizes"] != sizes():
        raise ValueError(f"{path} was trained with different cards or inputs; train a new model")
    net = Net()
    net.load_state_dict(saved["weights"])
    return net.eval()
