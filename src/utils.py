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


class PolicyOutput:
    def __init__(self, is_declare=None, declare_dict=None, to_ask=None, card=None):
        self.is_declare = is_declare
        self.declare_dict = declare_dict
        self.to_ask = to_ask
        self.card = card
