import random
from collections import defaultdict

import numpy as np

from policy import Policy
from utils import *


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

        self.card_tracker = torch.ones((n, deck_size), dtype=torch.int)
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
            raise ValueError()

        if self.cards[card] == -1:
            print(f"Card {card} already declared.")
            raise ValueError()

        requester = self.players[i]
        requested = self.players[j]

        # optimize these checks and add logging instead of print
        suit = get_suit(card)
        info = np.array([[i, j, card, 0]])

        if not any([suit == get_suit(own) for own in requester.cards]):
            # print(f"Player {i} does not have suit")
            self.turn = j
            raise ValueError()
        if card in requester.cards:
            print(f"Player {i} already has card")
            self.turn = j
            raise ValueError()

        toReturn = FAILS
        if self.cards[card] == j:
            requester.cards.add(card)
            requested.cards.remove(card)
            self.cards[card] = i
            info[:, 3] = 1
            toReturn = SUCCEEDS
            self.card_tracker[:, card] = 0
            self.card_tracker[i, card] = 1
            self.positive_asks[i] += 1
        else:
            self.turn = j
            self.card_tracker[i, card] = 0
            self.card_tracker[j, card] = 0
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
                raise ValueError()

        if len(declare_dict) != num_in_suit:
            print(f"Must declare exactly {num_in_suit} cards.")
            raise ValueError()

        if suit in self.declared_suites:
            print(f"Suit already declared")
            raise ValueError()

        # validate team
        teammates = set(declare_dict.values())
        declare_team = i < self.n // 2
        same_team = [not (declare_team ^ (teammate < self.n // 2)) for teammate in teammates]
        if not all(same_team):
            print(f"Declaration between different teams not allowed")
            raise ValueError()

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

    def step(self, policies):
        # want to print reward and action taken
        i = self.turn
        action = policies[i].ask(self)
        reward_dict = defaultdict(int)
        reward_dict[i] += self.asks(action.to_ask, action.card)

        for i, policy in enumerate(policies):
            action = policy.declare(self)
            while action is not None:
                action = policy.declare(self)
                reward_dict[i] += self.declare(action.declare_dict)

        # print(reward, action)
        self.n_rounds += 1
        if self.turn == i and len(self.players[i].cards) == 0 and not self.is_over():
            team = self.players[i].team
            same_team_with_cards = [j for j in range(team * (self.n // 2), (team + 1) * (self.n // 2)) if
                                    len(self.players[j].cards) > 0]

            if len(same_team_with_cards) > 0:
                self.turn = random.choice(same_team_with_cards)
            else:
                team = 1 - team
                other_team_with_cards = [j for j in range(team * (self.n // 2), (team + 1) * (self.n // 2)) if
                                         len(self.players[j].cards) > 0]
                self.turn = random.choice(other_team_with_cards)

        return reward_dict, action


def core_gameplay_loop(game, policies):
    while not game.is_over():
        policy = policies[game.turn]
        r, sp = game.step(policy)


if __name__ == "__main__":
    num_players = 6
    policies = [Policy() for i in range(num_players)]
    game = Game(num_players)
    core_gameplay_loop(game, policies)
