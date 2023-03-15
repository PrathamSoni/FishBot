import random
from utils import PolicyOutput, cards_of_same_suit


class Policy:
    def choose(self, game) -> PolicyOutput:
        pass


class RandomPolicy(Policy):
    def __init__(self):
        self.decide_threshold = .05

    def choose(self, game):
        if game.turn >= game.n // 2:
            r = range(game.n // 2)
        else:
            r = range(game.n // 2, game.n)
        action = PolicyOutput()
        if random.random() < self.decide_threshold or len(list(game.players[game.turn].cards)) == 0:
            action.is_declare = True
            card_choose = random.choice([x for x in game.cards if game.cards[x]>-1])
            to_declare = cards_of_same_suit(card_choose)
            team_offset = 0 if game.turn < game.n // 2 else 3
            action.declare_dict = {d : (team_offset + random.randint(0, 2)) for d in to_declare}
        else:
            action.is_declare = False
            action.to_ask = random.choice(r)
            cards_in_hand = list(game.players[game.turn].cards)
            card_choose = random.choice(cards_in_hand)
            action.card = random.choice(cards_of_same_suit(card_choose))
        return action
