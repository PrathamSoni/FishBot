import random
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

def expert_actions(iam, mycards, matrix):
    mat = torch.tensor(matrix)
    iamvec = torch.nn.functional.one_hot(torch.tensor([iam]), 6).flatten().bool()
    mat[iam] = torch.tensor([1 if c in mycards else 0 for c in range(54)])
    for c in range(54):
        if mat[iam][c] == 1:
            mat[~iamvec, c] = 0

    enemyteam = range(0,3) if iam >= 3 else range(3, 6)
    actions = []
    
    for suit in range(9):
        known = 0
        for cIdx in range(6):
            if sum(mat[:, 6 * suit + cIdx]) == 1:
                known += 1
        if known == 6:
            dec = PolicyOutput(is_declare=True, declare_dict={}, score=None)
            for cIdx in range(6):
                for p in range(6):
                    if mat[p, 6 * suit + cIdx] == 1:
                        if p in enemyteam:
                            #actions.append(PolicyOutput(is_declare=False, to_ask=p, card=6 * suit + cIdx))
                            return PolicyOutput(is_declare=False, to_ask=p, card=6 * suit + cIdx, score=None)
                            #dec.declare_dict.add(6 * suit + cIdx : iam)
                        else:
                            dec.declare_dict.update({6 * suit + cIdx : p})
                        break
            return dec
            actions.append(dec)
    return None

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
