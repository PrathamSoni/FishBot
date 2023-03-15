import random

import numpy as np

from policy import Policy
from utils import deck_size, num_in_suit, get_suit, deal

ILLEGAL = -10000
FAILS = -1
SUCCEEDS = 1
GOOD_DECLARE = 10
BAD_DECLARE = -10


class Player:
    def __init__(self, i, cards):
        self.team = None
        self.i = i
        self.cards = cards

    def set_team(self, team):
        self.team = team

    def __repr__(self):
        return "Player " + str(self.i) + ": \nTeam: " + str(self.team) + "\nCards: " + str(sorted(self.cards))


class Game:
    def __init__(self, n):
        self.n = n
        cards = deal()
        self.players = []
        cards_pp = deck_size // n

        self.cards = {}
        for i in range(n):
            hand = set(cards[i * cards_pp:(i + 1) * cards_pp])
            self.players.append(Player(i, hand))
            self.cards.update({card: i for card in hand})

        for player in self.players[:n // 2]:
            player.set_team(0)

        for player in self.players[n // 2:]:
            player.set_team(1)

        self.turn = random.choice(range(n))
        self.history = np.zeros((0, 4))

        self.card_tracker = np.zeros((n, deck_size))
        self.cumulative_reward = 0
        # Tracks players' asks and declares
        self.positive_asks = [0] * 6
        self.negative_asks = [0] * 6
        self.positive_declares = [0] * 6
        self.negative_declares = [0] * 6

        self.n_rounds = 0

        self.score = 0
        self.declared_suites = set()

    def __repr__(self):
        rep = [f"It's player {self.turn}'s turn!", f"The score is {self.score}.",
               f"Suits {self.declared_suites} have already been declared"] + [repr(self.players[i]) for i in
                                                                              range(self.n)]
        return "\n".join(rep)

    def asks(self, j, card):
        i = self.turn
        # print(f"Player {i} asked Player {j} for {card}")

        if not ((i < self.n // 2) ^ (j < self.n // 2)):
            print(f"Player {i} and Player {j} are on the same team.")
            return ILLEGAL

        if self.cards[card] == -1:
            print(f"Card {card} already declared.")
            return ILLEGAL

        requester = self.players[i]
        requested = self.players[j]

        # optimize these checks and add logging instead of print
        suit = get_suit(card)
        info = np.array([[i, j, card, 0]])

        if not any([suit == get_suit(own) for own in requester.cards]):
            # print(f"Player {i} does not have suit")
            self.turn = j
            return ILLEGAL
        if card in requester.cards:
            print(f"Player {i} already has card")
            self.turn = j
            return ILLEGAL

        toReturn = FAILS
        if self.cards[card] == j:
            requester.cards.add(card)
            requested.cards.remove(card)
            self.cards[card] = i
            info[:, 3] = 1
            toReturn = SUCCEEDS
            self.card_tracker[i, card] = 1
            self.positive_asks[i] += 1
        else:
            self.turn = j
            self.card_tracker[j, card] = -1
            self.negative_asks[i] += 1


        # print(f"Info this turn: {self.card_tracker}")
        self.history = np.concatenate([self.history, info])
        return toReturn

    def declare(self, declare_dict):
        i = self.turn

        # validate cards
        suit = None
        for card in declare_dict.keys():
            if suit is None:
                suit = get_suit(card)
                # print(f"Player {i} is declaring {suit}.")
            elif get_suit(card) != suit:
                print("Not all cards in same suit")
                return ILLEGAL

        if len(declare_dict) != num_in_suit:
            print(f"Must declare exactly {num_in_suit} cards.")
            return ILLEGAL

        if suit in self.declared_suites:
            print(self)
            print(f"Suit already declared")
            raise ValueError()

            return ILLEGAL

        # validate team
        teammates = set(declare_dict.values())
        declare_team = i < self.n // 2
        same_team = [not (declare_team ^ (teammate < self.n // 2)) for teammate in teammates]
        if not all(same_team):
            print(f"Declaration between different teams not allowed")
            return ILLEGAL

        correct = True
        for card, owner in declare_dict.items():
            true_owner = self.cards[card]
            if true_owner != owner:
                correct = False
                break

        if not (correct ^ declare_team):
            self.score += 1
        else:
            self.score -= 1

        for card in declare_dict.keys():
            true_owner = self.cards[card]
            self.players[true_owner].cards.remove(card)
            self.cards[card] = -1

        self.declared_suites.add(suit)
        if correct:
            self.positive_declares[i] += 1
        else:
            self.negative_declares[i] += 1
        return GOOD_DECLARE if correct else BAD_DECLARE

    def is_over(self):
        for p in self.players:
            if len(p.cards) != 0:
                return False
        return True

    def step(self, policy):
        # want to print reward and action taken
        action = policy.choose(self)
        if action.is_declare:
            reward = self.declare(action.declare_dict)
        else:
            reward = self.asks(action.to_ask, action.card)
        # print(reward, action)
        self.cumulative_reward += reward
        self.n_rounds += 1

        return reward, action


def core_gameplay_loop(game, policies):
    while not game.is_over():
        policy = policies[game.turn]
        r, sp = game.step(policy)


if __name__ == "__main__":
    num_players = 6
    policies = [Policy() for i in range(num_players)]
    game = Game(num_players)
    core_gameplay_loop(game, policies)
