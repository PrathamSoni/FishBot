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
