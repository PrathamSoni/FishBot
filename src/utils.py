import random

deck_size = 54
num_suits = 9
num_in_suit = deck_size // num_suits


def deal():
    deck = list(range(deck_size))
    random.shuffle(deck)
    return deck


def get_suit(card):
    return card // num_in_suit


def cards_of_suit(suit):
    return range(suit * num_in_suit, (suit + 1) * num_in_suit)


def suit_splice(suit):
    return slice(suit * num_in_suit, (suit + 1) * num_in_suit)


def cards_of_same_suit(card):
    return cards_of_suit(get_suit(card))


def get_suits_hand(hand):
    return set([get_suit(card) for card in hand])


class PolicyOutput:
    def __init__(self, is_declare=None, declare_dict=None, to_ask=None, card=None, score=None):
        self.is_declare = is_declare
        self.declare_dict = declare_dict
        self.to_ask = to_ask
        self.card = card
        self.score = score

    def __repr__(self):
        if self.declare_dict:
            return str(self.declare_dict)
        else:
            return str([self.to_ask, self.card])
