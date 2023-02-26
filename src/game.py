import random

deck_size = 54
num_suites = 9


class Player:
    def __init__(self, i, cards):
        self.team = None
        self.i = i
        self.cards = cards

    def set_team(self, team):
        self.team = team

    def __repr__(self):
        return "Player " + str(self.i) + ": \nTeam: " + str(self.team) + "\nCards: " + str(self.cards)


def deal():
    deck = list(range(deck_size))
    random.shuffle(deck)
    return deck


def get_suite(card):
    return card // (deck_size // num_suites)


class Game:
    def __init__(self, n):
        self.n = n
        cards = deal()
        self.players = []
        cards_pp = deck_size // n
        for i in range(n):
            self.players.append(Player(i, set(cards[i * cards_pp:(i + 1) * cards_pp])))

        for player in self.players[:n // 2]:
            player.set_team(0)

        for player in self.players[n // 2:]:
            player.set_team(1)

        self.turn = random.choice(range(n))

    def __repr__(self):
        rep = [f"It's player {self.turn}'s turn!"] + [repr(self.players[i]) for i in range(self.n)]
        return "\n".join(rep)

    def move(self, i, j, card):
        if self.turn != i:
            print(f"Not Player {i} turn.")
            return
        requester = self.players[i]
        requested = self.players[j]

        print(f"Player {i} asked Player {j} for {card}")

        # optimize these checks and add logging
        suite = get_suite(card)
        if not any([suite == get_suite(own) for own in requester.cards]):
            print(f"Player {i} does not have suite")
            self.turn = j
            return
        if card in requester.cards:
            print(f"Player {i} already has card")
            self.turn = j
            return

        if card in requested.cards:
            requester.cards.add(card)
            requested.cards.remove(card)
        else:
            self.turn = j


if __name__ == "__main__":
    game = Game(6)
    print(repr(game))
