import torch
from torch.nn import LSTM, Module, Embedding, Linear, BatchNorm1d

from utils import *


class RecurrentPlayer(Module):
    def __init__(self, i, embedding_dim=10, n_players=6):
        super().__init__()
        self.i = torch.tensor([i])
        self.embedding_dim = embedding_dim
        self.n_players = n_players

        self.cards_embedding = Embedding(deck_size, embedding_dim)
        self.players_embedding = Embedding(n_players, embedding_dim)

        self.hidden_dims = 2 * embedding_dim + 2

        self.final_embedding_layer = Linear(self.hidden_dims, self.hidden_dims)

        self.asking_player_layer = Linear(self.hidden_dims, n_players // 2)
        self.declaring_player_layer = Linear(self.hidden_dims, n_players // 2)
        self.asking_cards_layer = Linear(self.hidden_dims, deck_size)
        self.declaring_cards_layer = Linear(self.hidden_dims, deck_size)
        self.suit_scorer = Linear(num_in_suit, 1)

    def generate_embedding(self, score, cards):
        if cards.shape[0] == 0:
            own_cards = torch.zeros(self.embedding_dim)
        else:
            own_cards = self.cards_embedding(cards).sum(dim=0)

        final_embedding = torch.relu(torch.concatenate([own_cards, score, self.i]))
        return final_embedding

    def forward(self, score, history, cards, declared_suits):
        n, _ = history.shape
        final_embedding = self.generate_embedding(score, history, cards)
        asking_cards_pred = self.asking_cards_layer(final_embedding)
        declaring_cards_pred = self.declaring_cards_layer(final_embedding)

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
                print(cards, declared_suits, ask, ask_matrix)
            if self.i < self.n_players // 2:
                ask[0] += self.n_players // 2
            return False, ask, ask_score

    def choose(self, game):
        score = torch.tensor([game.score], dtype=torch.long)
        cards = torch.tensor(list(game.players[self.i].cards))
        declared_suits = torch.tensor(list(game.declared_suites), dtype=torch.long)

        declare, pred, score = self.forward(score, cards, declared_suits)

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
        self.policy_net = RecurrentPlayer(i, **kwargs)
        self.target_net = RecurrentPlayer(i, **kwargs)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.tau = tau

    def update_target(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                    1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
