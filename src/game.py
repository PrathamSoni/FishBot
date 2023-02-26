import random

deck_size = 54


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


class Game:
    def __init__(self, n):
        self.n = n
        cards = deal()
        self.players = []
        cards_pp = deck_size // n
        for i in range(n):
            self.players.append(Player(i, cards[i * cards_pp:(i + 1) * cards_pp]))

        for player in self.players[:n // 2]:
            player.set_team(0)

        for player in self.players[n // 2:]:
            player.set_team(1)

    def __repr__(self):
        rep = [repr(self.players[i]) for i in range(self.n)]
        return "\n".join(rep)


if __name__ == "__main__":
    game = Game(6)
    print(repr(game))
