# FishBot

Canadian Fish is a turn-based, information-gathering game played as 2 teams of 3 individuals. The game is played with a 54-card deck, consisting of the standard 52 cards plus two jokers (distinguishable). The 54 cards are organized into 9 half-suits, consisting of the groupings of 2-7 and 9-A within each suit across the four suits, and the final suit consisting of the 8's and jokers. The team to "declare" the most half-suits wins. 

To begin the game, the cards are shuffled and cards are dealt evenly amongst all players. An arbitrary player takes the first turn. In any given turn, a player asks a player on the opposing team for a card. The card asked for must be in a half-suit held by the player. If they ask for a card that the other player has, the other player must give them the card. This player continues to ask for cards until they request a card incorrectly. The player is able to change to whom they are asking for a card between asks given that the previous asks were all successful. To declare, a player announces a half-suit to declare and enumerates which cards from their half-suit each player on their team has. Notably, all cards must be within that team and the enumeration must be correct; otherwise, the opposition team gets the half-suit. A player can also declare if they have all of the cards in their own hand. There is no restriction on the distribution of cards between team members, and a team can declare at any time. Finally, if a player declares while it is their turn, and they run out of cards in hand, they can choose which teammate to pass their turn onto.
```

from game import Game
from models import RecurrentPlayer
game = Game(6)
players = [RecurrentPlayer(i) for i in range(6)]
for i in range(10):
     print(game.is_over())
     if game.is_over():
        break
     print(game.turn)
     policy = players[game.turn]
     r, sp = game.step(policy)

```