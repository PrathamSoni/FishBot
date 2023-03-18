from torch.nn import Module, Sequential, ReLU, Linear
import torch.nn.functional as F

from utils import *


class MoveEval(Module):
    def __init__(self, hidden_dim=64, n_players=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_players = n_players

        self.layers = Sequential(Linear(3 * n_players // 2 * num_in_suit * num_suits, hidden_dim), ReLU(),
                                 Linear(hidden_dim, hidden_dim), ReLU(),
                                 Linear(hidden_dim, 1))
        self.history = []

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

    def forward(self, move):
        return self.layers(move)

    def ask(self, game):
        cards = game.players[game.turn].cards
        moves = valid_asks(game.turn, cards, game.card_tracker)

        m, _ = moves.shape
        moves = torch.cat([game.card_tracker.flatten().expand(m, game.n * num_in_suit * num_suits), moves], dim=-1)
        scores = self.forward(moves)
        if m==0:
            import pdb; pdb.set_trace()
        score = scores.max()
        move = moves[scores.argmax()]

        self.history.append(move)
        move = move[-game.n // 2 * num_in_suit * num_suits:]

        coordinate = move.argmax().item()
        player, card = divmod(coordinate, num_suits * num_in_suit)

        if game.players[game.turn].team == 0:
            player += game.n // 2

        return PolicyOutput(
            is_declare=False,
            to_ask=player,
            card=card,
            score=score,
            player=game.turn,
        )

    def declare(self, game):
        all_declares = []
        for player in range(self.n_players):
            cards = game.players[player].cards
            declares = valid_declares(player, cards, game.card_tracker)
            all_declares.extend([PolicyOutput(
                is_declare=True,
                declare_dict=valid_declare,
                score=GOOD_DECLARE,
                player=player
            ) for valid_declare in declares])

        return all_declares
