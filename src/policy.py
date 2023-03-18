import random

from utils import GOOD_DECLARE, PolicyOutput, cards_of_same_suit, get_suit, get_suits_hand, cards_of_suit


class Policy:
    def ask(self, game) -> PolicyOutput:
        pass

    def declare(self, game) -> list[PolicyOutput]:
        pass


class RandomPolicy(Policy):
    def __init__(self):
        self.declare_threshold = .01

    # def choose(self, game):
    #     if game.turn >= game.n // 2:
    #         r = range(game.n // 2)
    #     else:
    #         r = range(game.n // 2, game.n)

    #     cards_in_hand = game.players[game.turn].cards
    #     suits_in_hand = get_suits_hand(cards_in_hand)
    #     cards_of_suits = set([card for suit in suits_in_hand for card in list(cards_of_suit(suit))])
    #     cards = list(cards_of_suits - cards_in_hand)

    #     action = PolicyOutput(is_declare=False, to_ask=None, score=None)
    #     if random.random() < self.declare_threshold or len(list(game.players[game.turn].cards)) == 0 or len(cards) == 0:
    #         action.is_declare = True
    #         card_choose = random.choice([x for x in game.cards if game.cards[x] > -1])
    #         to_declare = cards_of_same_suit(card_choose)
    #         team_offset = 0 if game.turn < game.n // 2 else 3
    #         action.declare_dict = {d: (team_offset + random.randint(0, 2)) for d in to_declare}
    #     else:
    #         action.is_declare = False
    #         action.to_ask = random.choice(r)
    #         action.card = random.choice(cards)
    #     return action

    def get_cards(self, game, player):
        cards_in_hand = game.players[player].cards
        suits_in_hand = get_suits_hand(cards_in_hand)
        cards_of_suits = set([card for suit in suits_in_hand for card in list(cards_of_suit(suit))])
        cards = list(cards_of_suits - cards_in_hand)
        return cards

    def ask(self, game) -> PolicyOutput:
        # ask for a random card from a random player
        if game.turn >= game.n // 2:
            r = range(game.n // 2)
        else:
            r = range(game.n // 2, game.n)
        cards = self.get_cards(game, game.turn)
        to_ask = random.choice(r)
        if not to_ask:
            return PolicyOutput(is_declare=False, to_ask=None, card=None, score=None, player=game.turn)
        else:
            card = random.choice(cards)
            return PolicyOutput(is_declare=False, to_ask=to_ask, card=card, score=None, player=game.turn)

    def declare(self, game) -> list[PolicyOutput]:
        # output list of PolicyOutputs
        actions = []
        for player in range(game.n):
            cards = self.get_cards(game, player)
            if random.random() < self.declare_threshold or len(list(game.players[player].cards)) == 0 or len(cards) == 0:
                if game.is_over():
                    break
                
                card_choose = random.choice([x for x in game.cards if (game.cards[x] > -1)])

                to_declare = cards_of_same_suit(card_choose)
                if get_suit(card_choose) not in game.declared_suites:
                    team_offset = 0 if player < game.n // 2 else 3
                    action = PolicyOutput(is_declare=True, to_ask=None, score=None, player=player)
                    action.declare_dict = {d: (team_offset + random.randint(0, 2)) for d in to_declare}
                    actions.append(action)
                    break
            
        return actions