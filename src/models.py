import random

import torch
from torch.nn import LSTM, Module, Embedding, Linear

from game import Game
from utils import deck_size, num_suits, num_in_suit, suit_splice, get_suits_hand, PolicyOutput, cards_of_suit


class RecurrentPlayer(Module):
    def __init__(self, i, embedding_dim=10, n_players=6, declare_threshold=.99):
        super().__init__()
        self.i = torch.tensor([i])
        self.embedding_dim = embedding_dim
        self.n_players = n_players

        self.cards_embedding = Embedding(deck_size, embedding_dim)
        self.players_embedding = Embedding(n_players, embedding_dim)
        self.history_embedding = LSTM(3 * embedding_dim + 1, embedding_dim)

        self.hidden_dims = 2 * embedding_dim + 2

        self.asking_player_layer = Linear(self.hidden_dims, n_players // 2)
        self.declaring_player_layer = Linear(self.hidden_dims, n_players // 2)
        self.asking_cards_layer = Linear(self.hidden_dims, deck_size)
        self.declaring_cards_layer = Linear(self.hidden_dims, deck_size)

        self.declare_threshold = declare_threshold

    def generate_embedding(self, score, history, cards):
        if cards.shape[0] == 0:
            own_cards = torch.zeros(self.embedding_dim)
        else:
            own_cards = self.cards_embedding(cards).sum(dim=0)

        asking_players = history[:, 0]
        asked_players = history[:, 1]
        asked_cards = history[:, 2]
        success = history[:, 3].unsqueeze(-1)

        asking_players_embedding = self.players_embedding(asking_players)
        asked_players_embedding = self.players_embedding(asked_players)
        asked_cards_embedding = self.cards_embedding(asked_cards)
        history_input = torch.concatenate(
            [asking_players_embedding, asked_players_embedding, asked_cards_embedding, success], dim=1)

        if history_input.shape[0] == 0:
            history_features = torch.zeros(self.embedding_dim)
        else:
            history_features = self.history_embedding(history_input)[0][-1]

        final_embedding = torch.concatenate([own_cards, history_features, score, self.i])
        return final_embedding

    def forward(self, score, history, cards, declared_suits):
        n, _ = history.shape
        final_embedding = self.generate_embedding(score, history, cards)
        asking_cards_pred = self.asking_cards_layer(final_embedding)
        declaring_cards_pred = self.declaring_cards_layer(final_embedding)

        asking_player_pred = self.asking_player_layer(final_embedding)
        ask_matrix = asking_player_pred.unsqueeze(0).T @ asking_cards_pred.unsqueeze(0)

        hand = cards.tolist()
        allowable = [cards_of_suit(suit) for suit in get_suits_hand(hand)]
        allowable = set([item for sublist in allowable for item in sublist])
        not_allowable = list(set(range(deck_size)) - allowable)
        ask_matrix[:, hand] = -torch.inf
        ask_matrix[:, not_allowable] = -torch.inf
        ask_matrix = torch.sigmoid(ask_matrix)
        ask_matrix = ask_matrix / ask_matrix.sum()
        ask_score = ask_matrix.max() * 2 - 1
        ask = (ask_matrix == torch.max(ask_matrix)).nonzero().squeeze().tolist()

        declaring_player_pred = self.declaring_player_layer(final_embedding)
        declare_matrix = declaring_player_pred.unsqueeze(0).T @ declaring_cards_pred.unsqueeze(0)
        player_mod = self.i % (self.n_players // 2)
        declare_matrix[player_mod, hand] = torch.inf
        other_players = torch.tensor([i for i in range(self.n_players // 2) if i != player_mod]).unsqueeze(-1)
        declare_matrix[other_players, hand] = -torch.inf

        declare_matrix = declare_matrix.reshape((self.n_players // 2, num_in_suit, num_suits))
        declare_matrix[:, :, declared_suits] = -torch.inf
        score_matrix, args = declare_matrix.max(dim=0)
        suit_scores = torch.sigmoid(score_matrix)
        suit_scores = suit_scores / suit_scores.sum()
        suit_scores = suit_scores.prod(dim=0)
        suit = suit_scores.argmax().item()
        declare_score = torch.pow(suit_scores.max(), 25 / (n + 1)) * 2 - 1
        owners = args[:, suit]

        # if no cards you must declare
        if cards.shape[0] == 0 or declare_score > ask_score or len(ask) == 0:
            if suit in declared_suits:
                print(suit, declared_suits, cards, declare_matrix.reshape(3, 9, 6))
                raise ValueError()
            if self.i >= self.n_players // 2:
                owners = owners + self.n_players // 2

            return True, {card: owner for card, owner in
                          zip(range(suit * num_in_suit, (suit + 1) * num_in_suit),
                              owners.tolist())}, declare_score * 10

        else:
            try:
                if type(ask[0]) == list:
                    ask = random.choice(ask)
            except:
                print(history, cards, declared_suits, ask, ask_matrix)
            if self.i < self.n_players // 2:
                ask[0] += self.n_players // 2
            return False, ask, ask_score

    def choose(self, game):
        score = torch.tensor([game.score], dtype=torch.long)
        history = torch.tensor(game.history, dtype=torch.long)
        cards = torch.tensor(list(game.players[self.i].cards))
        declared_suits = torch.tensor(list(game.declared_suites), dtype=torch.long)
        n_rounds = game.n_rounds

        declare, pred, score = self.forward(score, history, cards, n_rounds, declared_suits)

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


class RecurrentTrainer:
    def __init__(self, i, tau=.005, *args, **kwargs):
        self.policy_net = RecurrentPlayer(i, *args, **kwargs)
        self.target_net = RecurrentPlayer(i, *args, **kwargs)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.tau = tau

    def update_target(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                    1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


class RecurrentPlayer2(Module):
    def __init__(self, i, embedding_dim=10, n_players=6, declare_threshold=.99):
        super().__init__()
        self.i = torch.tensor([i])
        self.embedding_dim = embedding_dim
        self.n_players = n_players

        self.cards_embedding = Embedding(deck_size, embedding_dim)
        self.players_embedding = Embedding(n_players, embedding_dim)
        self.card_tracker_emb = LSTM(embedding_dim, embedding_dim)

        self.card_status_embedding = Embedding(deck_size * n_players, embedding_dim)

        self.hidden_dims = 2 * embedding_dim + 2

        self.asking_player_layer = Linear(self.hidden_dims, n_players // 2)
        self.declaring_player_layer = Linear(self.hidden_dims, n_players // 2)
        self.asking_cards_layer = Linear(self.hidden_dims, deck_size)
        self.declaring_cards_layer = Linear(self.hidden_dims, deck_size)

        self.declare_threshold = declare_threshold

    def generate_embedding(self, score, card_tracker, cards):
        if cards.shape[0] == 0:
            own_cards = torch.zeros(self.embedding_dim)
        else:
            own_cards = self.cards_embedding(cards).sum(dim=0)

        # Fixing indexing (hack)
        card_tracker[card_tracker == -1] = 300
        # essentially encodes asking player, asked player, asked cards all in one.
        card_status_embedding = self.card_status_embedding(card_tracker)
        card_tracker_input = torch.concatenate(
            [card_status_embedding], dim=1)

        if card_tracker_input.shape[0] == 0:
            card_tracker_features = torch.zeros(self.embedding_dim)
        else:
            card_tracker_features = self.card_tracker_emb(card_tracker_input)[0][-1]

        final_embedding = torch.concatenate([own_cards, card_tracker_features, score, self.i])
        return final_embedding

    def forward(self, score, card_tracker, n_rounds, cards, declared_suits):
        
        final_embedding = self.generate_embedding(score, card_tracker, cards)
        asking_cards_pred = self.asking_cards_layer(final_embedding)
        declaring_cards_pred = self.declaring_cards_layer(final_embedding)

        asking_player_pred = self.asking_player_layer(final_embedding)
        ask_matrix = asking_player_pred.unsqueeze(0).T @ asking_cards_pred.unsqueeze(0)

        hand = cards.tolist()
        allowable = [cards_of_suit(suit) for suit in get_suits_hand(hand)]
        allowable = set([item for sublist in allowable for item in sublist])
        not_allowable = list(set(range(deck_size)) - allowable)
        ask_matrix[:, hand] = -torch.inf
        ask_matrix[:, not_allowable] = -torch.inf
        ask_matrix = torch.sigmoid(ask_matrix)
        ask_matrix = ask_matrix / ask_matrix.sum()
        ask_score = ask_matrix.max() * 2 - 1
        ask = (ask_matrix == torch.max(ask_matrix)).nonzero().squeeze().tolist()

        declaring_player_pred = self.declaring_player_layer(final_embedding)
        declare_matrix = declaring_player_pred.unsqueeze(0).T @ declaring_cards_pred.unsqueeze(0)
        player_mod = self.i % (self.n_players // 2)
        declare_matrix[player_mod, hand] = torch.inf
        other_players = torch.tensor([i for i in range(self.n_players // 2) if i != player_mod]).unsqueeze(-1)
        declare_matrix[other_players, hand] = -torch.inf

        declare_matrix = declare_matrix.reshape((self.n_players // 2, num_in_suit, num_suits))
        declare_matrix[:, :, declared_suits] = -torch.inf
        score_matrix, args = declare_matrix.max(dim=0)
        suit_scores = torch.sigmoid(score_matrix)
        suit_scores = suit_scores / suit_scores.sum()
        suit_scores = suit_scores.prod(dim=0)
        suit = suit_scores.argmax().item()

        declare_score = torch.pow(suit_scores.max(), 50 / (n_rounds + 1)) * 2 - 1
        owners = args[:, suit]

        # if no cards you must declare
        if cards.shape[0] == 0 or declare_score > ask_score or len(ask) == 0:
            if suit in declared_suits:
                print(suit, declared_suits, cards, declare_matrix.reshape(3, 9, 6))
                raise ValueError()

            if self.i >= self.n_players // 2:
                owners = owners + self.n_players // 2

            return True, {card: owner for card, owner in
                          zip(range(suit * num_in_suit, (suit + 1) * num_in_suit),
                              owners.tolist())}, declare_score * 10

        else:
            try:
                if type(ask[0]) == list:
                    ask = random.choice(ask)
            except:
                print(history, cards, declared_suits, ask, ask_matrix.reshape(3, 9, 6))
            if self.i < self.n_players // 2:
                ask[0] += self.n_players // 2
            return False, ask, ask_score

    def choose(self, game):
        score = torch.tensor([game.score], dtype=torch.long)
        cards = torch.tensor(list(game.players[self.i].cards))
        declared_suits = torch.tensor(list(game.declared_suites), dtype=torch.long)
        card_tracker = torch.tensor(game.card_tracker, dtype=torch.long).flatten()
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


if __name__ == "__main__":
    game = Game(6)
    players = [RecurrentPlayer(i) for i in range(6)]
    while not game.is_over():
        if game.is_over():
            break
        policy = players[game.turn]
        r, sp = game.step(policy)
