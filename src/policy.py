import random
from utils import PolicyOutput, cards_of_same_suit


class Policy:
    def choose(self, game) -> PolicyOutput:
        pass


class RandomPolicy(Policy):
    def __init__(self):
        self.decide_threshold = .2

    def choose(self, game):
        if self.turn < game.n / 2:
            r = range(game.n / 2)
        else:
            r = range(game.n / 2, game.n)
        action = PolicyOutput()
        if random.random() < self.decide_threshold:
            action.is_declare = True
            action.declare_dict = [random.randint(0, 2) for i in r]
        else:
            action.is_declare = False
            action.to_ask = random.choice(r)
            cards_in_hand = game.players[game.turn].cards
            card_choose = random.choice(cards_in_hand)
            action.card = random.choice(cards_of_same_suit(card_choose))
        return action
