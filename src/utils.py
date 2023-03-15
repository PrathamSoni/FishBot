import random

deck_size = 54
num_suits = 9
num_in_suit = deck_size // num_suits

CONVERT_DICT = {0 : [0, 1, 2, 3, 4, 5],
                1 : [1, 0, 2, 3, 4, 5],
                2 : [2, 0, 1, 3, 4, 5],
                3 : [3, 4, 5, 0, 1, 2],
                4 : [4, 3, 5, 0, 1, 2],
                5 : [5, 3, 4, 0, 1, 2],
                }


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

def convert(iam, heis):
    return CONVERT_DICT[iam][heis]


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
