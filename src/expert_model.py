import random

import torch
from torch.nn import Module, Linear
import torch.nn.functional as F

from utils import *


class RecurrentPlayer2(Module):
    def __init__(self, i, embedding_dim=512, n_players=6):
        super().__init__()
        self.i = torch.tensor([i])
        self.embedding_dim = embedding_dim
        self.n_players = n_players

        self.card_tracker_emb = Linear(deck_size * n_players, embedding_dim)

        self.hidden_dims = embedding_dim + 1
        self.final_embedding_layer = Linear(self.hidden_dims, self.hidden_dims)

        self.asking_player_layer = Linear(self.hidden_dims, n_players // 2)
        self.declaring_player_layer = Linear(self.hidden_dims, n_players // 2)
        self.asking_cards_layer = Linear(self.hidden_dims, deck_size)
        self.declaring_cards_layer = Linear(self.hidden_dims, deck_size)
        self.suit_scorer = Linear(num_in_suit, 1)

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

    def forward(self, score, card_tracker, n_rounds, cards, declared_suits):

        final_embedding = self.generate_embedding(score, card_tracker, cards)
        final_embedding = torch.relu(self.final_embedding_layer(final_embedding))

        asking_cards_pred = torch.tanh(self.asking_cards_layer(final_embedding))
        asking_player_pred = torch.tanh(self.asking_player_layer(final_embedding))
        ask_matrix = asking_player_pred.unsqueeze(0).T @ asking_cards_pred.unsqueeze(0)
        ask_matrix = normalize(ask_matrix) * SUCCEEDS

        hand = cards.tolist()
        allowable = [cards_of_suit(suit) for suit in get_suits_hand(hand)]
        allowable = set([item for sublist in allowable for item in sublist])
        not_allowable = list(set(range(deck_size)) - allowable)
        ask_matrix[:, hand] = -SUCCEEDS
        ask_matrix[:, not_allowable] = -SUCCEEDS
        ask_score = ask_matrix.max()
        ask = (ask_matrix == torch.max(ask_matrix)).nonzero().squeeze().tolist()

        declaring_cards_pred = torch.tanh((self.declaring_cards_layer(final_embedding)))
        declaring_player_pred = torch.tanh(self.declaring_player_layer(final_embedding))
        declare_matrix = declaring_player_pred.unsqueeze(0).T @ declaring_cards_pred.unsqueeze(0)  # raw score
        declare_matrix = normalize(declare_matrix)
        player_mod = self.i % (self.n_players // 2)
        declare_matrix[player_mod, hand] = 1
        other_players = torch.tensor([i for i in range(self.n_players // 2) if i != player_mod]).unsqueeze(-1)
        declare_matrix[other_players, hand] = -1

        declare_matrix = declare_matrix.reshape((self.n_players // 2, num_suits, num_in_suit))
        suit_scores, args = declare_matrix.max(dim=0)
        suit_scores = normalize(self.suit_scorer(suit_scores)) * GOOD_DECLARE
        suit_scores[declared_suits] = -GOOD_DECLARE

        if torch.all(torch.isnan(suit_scores)) or suit_scores.sum().item() == 0:
            suit = random.choice(list(set(range(num_suits)) - set(declared_suits.tolist())))
        else:
            suit = suit_scores.argmax().item()

        declare_score = suit_scores.max()
        owners = args[suit]
        # print(ask_score, declare_score)
        if torch.isnan(ask_score + declare_score):
            import pdb;
            pdb.set_trace()
        # if no cards you must declare
        if cards.shape[0] == 0 or declare_score > ask_score * GOOD_DECLARE / SUCCEEDS or len(ask) == 0:
            if suit in declared_suits:
                suit = random.choice(list(set(range(num_suits)) - set(declared_suits.tolist())))
                owners = args[suit]

            if self.i >= self.n_players // 2:
                owners = owners + self.n_players // 2

            return True, {card: owner for card, owner in
                          zip(range(suit * num_in_suit, (suit + 1) * num_in_suit),
                              owners.tolist())}, declare_score

        else:
            try:
                if type(ask[0]) == list:
                    ask = random.choice(ask)
            except:
                print(card_tracker, cards, declared_suits, ask, ask_matrix)
            if self.i < self.n_players // 2:
                ask[0] += self.n_players // 2
            return False, ask, ask_score

    def choose(self, game):
        score = torch.tensor([game.score], dtype=torch.long)
        cards = torch.tensor(list(game.players[self.i].cards))
        declared_suits = torch.tensor(list(game.declared_suites), dtype=torch.long)
        card_tracker = game.card_tracker
        n_rounds = game.n_rounds

        declare, pred, score = self.forward(score, card_tracker, n_rounds, cards, declared_suits)

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


class RecurrentTrainer2:
    def __init__(self, i, tau=.005, *args, **kwargs):
        self.policy_net = RecurrentPlayer2(i, *args, **kwargs)
        self.target_net = RecurrentPlayer2(i, *args, **kwargs)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.tau = tau

    def update_target(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                    1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
