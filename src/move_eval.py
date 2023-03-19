import random

from torch.nn import Module, Sequential, ReLU, Linear
import torch.nn.functional as F

from utils import *


class MoveEval(Module):
    def __init__(self, hidden_dim=64, n_players=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_players = n_players

        self.ask_layers = Sequential(Linear(3 * n_players // 2 * num_in_suit * num_suits + n_players, hidden_dim),
                                     ReLU(),
                                     Linear(hidden_dim, hidden_dim), ReLU(),
                                     Linear(hidden_dim, 1))
        self.declare_layers = Sequential(Linear(n_players * num_in_suit * num_suits + n_players + num_suits + n_players//2*num_in_suit, hidden_dim),
                                     ReLU(),
                                     Linear(hidden_dim, hidden_dim), ReLU(),
                                     Linear(hidden_dim, 1))
        self.ask_history = []
        self.declare_history = []

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

    def forward(self, move, type="ask"):
        if type == "ask":
            return self.ask_layers(move)
        elif type == "declare":
            return self.declare_layers(move)

    def ask(self, game):
        i = game.turn
        cards = game.players[i].cards
        card_tracker = game.card_tracker
        one_hot = torch.tensor([i in cards for i in range(num_in_suit * num_suits)])
        card_tracker = card_tracker * ~one_hot
        card_tracker[i] = one_hot
        moves = valid_asks(i, cards, card_tracker)

        m, _ = moves.shape
        cards_in_hand = torch.tensor([len(game.players[i].cards) for i in range(game.n)]).expand(m, game.n)
        moves = torch.cat([card_tracker.flatten().expand(m, game.n * num_in_suit * num_suits), cards_in_hand, moves],
                          dim=-1)
        scores = self(moves, "ask")
        if m == 0:
            import pdb;
            pdb.set_trace()
        score = scores.max()
        move = moves[scores.argmax()]

        self.ask_history.append(move)
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
            player=i,
        )

    def declare(self, game, i):
        cards = game.players[i].cards
        declares = valid_declares(i, cards, game.card_tracker)
        all_declares = []
        for suit, combos in declares.items():
            if len(combos) == 0:
                continue

            moves = torch.zeros((len(combos), num_suits + game.n//2*num_in_suit))
            moves[:, suit] = 1
            for j, combo in enumerate(combos):

                one_hot = [num_suits + j*game.n//2 + v for j, v in enumerate(combo)]
                moves[j, one_hot] = 1

            m, _ = moves.shape
            cards_in_hand = torch.tensor([len(game.players[i].cards) for i in range(game.n)]).expand(m, game.n)
            moves = torch.cat(
                [game.card_tracker.flatten().expand(m, game.n * num_in_suit * num_suits), cards_in_hand, moves],
                dim=-1)
            scores = self(moves, "declare")
            score = scores.max()
            best = scores.argmax().item()
            move = moves[best]

            self.declare_history.append(move)

            assignments = combos[best]
            if i >= 3:
                assignments = [c + 3 for c in assignments]

            all_declares.append(PolicyOutput(
                is_declare=True,
                declare_dict=dict(zip(cards_of_suit(suit), assignments)),
                score=score,
                player=i
            ))

        return all_declares
