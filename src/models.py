import torch
from torch.nn import LSTM, Module, Embedding, Linear

from game import Game
from utils import deck_size, num_suits, num_in_suit, suit_splice, get_suits_hand, PolicyOutput, cards_of_suit


class RecurrentPlayer(Module):
    def __init__(self, i, embedding_dim=10, n_players=6, declare_threshold=.9):
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
        self.cards_layer = Linear(self.hidden_dims, deck_size)

        self.declare_layer = Linear(self.hidden_dims, 1)
        self.suit_layer = Linear(self.hidden_dims, num_suits)

        self.declare_threshold = declare_threshold

    def forward(self, score, history, cards, declared_suits):
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
        cards_pred = self.cards_layer(final_embedding)

        declare_score = torch.sigmoid(self.declare_layer(final_embedding)).item()

        if declare_score < self.declare_threshold and cards.shape[0] != 0:
            player_pred = self.asking_player_layer(final_embedding)

            ask_matrix = player_pred.unsqueeze(0).T @ cards_pred.unsqueeze(0)
            ask_matrix = ask_matrix - ask_matrix.min()

            hand = cards.tolist()
            allowable = [cards_of_suit(suit) for suit in get_suits_hand(hand)]
            allowable = set([item for sublist in allowable for item in sublist])
            not_allowable = list(set(range(deck_size)) - allowable)
            ask_matrix[:, hand] = -1
            ask_matrix[:, not_allowable] = -1
            ask = (ask_matrix == torch.max(ask_matrix)).nonzero().squeeze().tolist()
            if self.i < self.n_players // 2:
                ask[0] += self.n_players // 2

            return declare_score, ask, ask_matrix.max()
        else:
            suit_pred = self.suit_layer(final_embedding)
            suit_pred = suit_pred - suit_pred.min()
            suit_pred[declared_suits] = -1
            suit = torch.argmax(suit_pred)

            player_pred = self.declaring_player_layer(final_embedding)
            card_filter = (cards[(suit * num_in_suit <= cards) & (
                        cards <= (suit + 1) * num_in_suit)] % num_in_suit).tolist()
            declare_matrix = player_pred.unsqueeze(0).T @ cards_pred[suit_splice(suit)].unsqueeze(0)

            player_mod = self.i % (self.n_players // 2)
            declare_matrix[player_mod, card_filter] = 1
            other_players = torch.tensor([i for i in range(self.n_players // 2) if i != player_mod]).unsqueeze(-1)
            declare_matrix[other_players, card_filter] = -1

            owners = torch.argmax(declare_matrix, dim=0)
            if self.i >= self.n_players // 2:
                owners = owners + self.n_players // 2

            return declare_score, {card: owner for card, owner in
                                   zip(range(suit * num_in_suit, (suit + 1) * num_in_suit),
                                       owners.tolist())}, suit_pred.max()

    def choose(self, game):
        score = torch.tensor([game.score], dtype=torch.long)
        history = torch.tensor(game.history, dtype=torch.long)
        cards = torch.tensor(list(game.players[self.i].cards))
        declared_suits = torch.tensor(list(game.declared_suites), dtype=torch.long)

        declare_score, pred, score = self.forward(score, history, cards, declared_suits)

        if declare_score < self.declare_threshold and cards.shape[0] != 0:
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


if __name__ == "__main__":
    game = Game(6)
    players = [RecurrentPlayer(i) for i in range(6)]
    while not game.is_over():
        if game.is_over():
            break
        policy = players[game.turn]
        r, sp = game.step(policy)
