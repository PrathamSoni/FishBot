import torch
from torch.nn import Module
import torch.nn.functional as F

from utils import *


class MoveEval(Module):
    def __init__(self, i, embedding_dim=512, n_players=6):
        super().__init__()
        self.i = torch.tensor([i])
        self.embedding_dim = embedding_dim
        self.n_players = n_players

    def generate_embedding(self, score, card_tracker, cards):
        tracker_clone = torch.tensor(card_tracker)
        idx = self.i.item()
        tracker_clone[idx] = torch.tensor([1e7 if c in cards else 0 for c in range(54)])
        tracker_clone = F.normalize(tracker_clone, p=1, dim=0)
        tracker_list = [tracker_clone[idx]]
        friend_range = range(self.n_players // 2) if idx < self.n_players / 2 else range(self.n_players // 2,
                                                                                         self.n_players)
        enemy_range = range(self.n_players // 2) if idx >= self.n_players / 2 else range(self.n_players // 2,
                                                                                         self.n_players)
        for i in friend_range:
            if i != idx:
                tracker_list.append(tracker_clone[i])
        for i in enemy_range:
            tracker_list.append(tracker_clone[i])
        tracker_input = torch.cat(tracker_list).float()
        # essentially encodes asking player, asked player, asked cards all in one.
        card_tracker_embedding = F.relu(self.card_tracker_emb(tracker_input))

        final_embedding = torch.concatenate([card_tracker_embedding, score])
        return final_embedding

    def forward(self, score, card_tracker, cards):
        final_embedding = self.generate_embedding(score, card_tracker, cards)

    def asks(self, game):
        score = torch.tensor([game.score], dtype=torch.long)
        cards = torch.tensor(list(game.players[self.i].cards))
        card_tracker = game.card_tracker

        declare, pred, score = self.forward(score, card_tracker, cards)

        if not declare:
            output = PolicyOutput(
                is_declare=False,
                to_ask=pred[0],
                card=pred[1],
                score=score
            )
        else:
            output = PolicyOutput(
                is_declare=True,
                declare_dict=pred,
                score=score
            )
        return output
