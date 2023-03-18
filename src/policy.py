import random
from utils import PolicyOutput, cards_of_same_suit, get_suits_hand, cards_of_suit


class Policy:
    def ask(self, game) -> PolicyOutput:
        pass

    def declare(self, game) -> list[PolicyOutput]:
        pass


class RandomPolicy(Policy):
    def __init__(self):
        self.declare_threshold = .05

    def choose(self, game):
        if game.turn >= game.n // 2:
            r = range(game.n // 2)
        else:
            r = range(game.n // 2, game.n)

        cards_in_hand = game.players[game.turn].cards
        suits_in_hand = get_suits_hand(cards_in_hand)
        cards_of_suits = set([card for suit in suits_in_hand for card in list(cards_of_suit(suit))])
        cards = list(cards_of_suits - cards_in_hand)

        action = PolicyOutput()
        if random.random() < self.declare_threshold or len(list(game.players[game.turn].cards)) == 0 or len(cards) == 0:
            action.is_declare = True
            card_choose = random.choice([x for x in game.cards if game.cards[x] > -1])
            to_declare = cards_of_same_suit(card_choose)
            team_offset = 0 if game.turn < game.n // 2 else 3
            action.declare_dict = {d: (team_offset + random.randint(0, 2)) for d in to_declare}
        else:
            action.is_declare = False
            action.to_ask = random.choice(r)
            action.card = random.choice(cards)
        return action

    def ask(self, game):
        pass

    def declare(self, game):
        pass
