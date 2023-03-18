import random
from dataclasses import dataclass
from typing import Optional

import torch

deck_size = 54
num_suits = 9
num_in_suit = deck_size // num_suits

CONVERT_DICT = {0: [0, 1, 2, 3, 4, 5],
                1: [1, 0, 2, 3, 4, 5],
                2: [2, 0, 1, 3, 4, 5],
                3: [3, 4, 5, 0, 1, 2],
                4: [4, 3, 5, 0, 1, 2],
                5: [5, 3, 4, 0, 1, 2],
                }

ILLEGAL = -10000
FAILS = -1
SUCCEEDS = 1
GOOD_DECLARE = 10
BAD_DECLARE = -10
WIN_GAME = 100
LOSE_GAME = -100
TEAM0 = torch.tensor([True, True, True, False, False, False], dtype=torch.bool)
EYE = torch.eye(3 * deck_size)


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


@dataclass
class PolicyOutput:
    is_declare: bool
    score: torch.Tensor
    declare_dict: Optional[dict] = None
    to_ask: Optional[int] = None
    card: Optional[int] = None
    success: Optional[bool] = None

    def __repr__(self):
        if self.is_declare:
            return str(self.declare_dict)
        else:
            return str([self.to_ask, self.card])


def normalize(score):
    return 2 * score / (score.max() - score.min()) - 1 - 2 * score.min() / (
            score.max() - score.min())

def _suits_mask(mycards):
    suits = get_suits_hand(mycards)
    return torch.tensor([int(i // num_in_suit in suits and i not in mycards) 
                         for i in range(deck_size)] * 3, dtype=torch.int)


def valid_asks(iam, mycards, matrix):
    enemies = TEAM0 if iam >= 3 else ~TEAM0
    suits_mask = _suits_mask(mycards)
    region_of_interest = matrix[enemies, :].flatten()
    askable = torch.mul(region_of_interest, suits_mask).bool()
    return EYE[askable]
