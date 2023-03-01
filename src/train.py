from game import Game
from models import RecurrentTrainer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def train(games, batch_size, gamma, tau, lr):
    n = 6
    trainers = [RecurrentTrainer(i) for i in range(n)]
    optimizers = [optim.AdamW(trainer.policy_net.parameters(), lr=LR, amsgrad=True) for trainer in trainers]

    def optimize():
        # print(losses)
        for i in range(n):
            if type(losses[i]) != int:
                optimizer = optimizers[i]
                optimizer.zero_grad()

                losses[i].backward()
                trainer = trainers[i]

                torch.nn.utils.clip_grad_value_(trainer.policy_net.parameters(), 100)
                optimizer.step()

                trainer.update_target()

    for g in range(games):
        print(f"Game {g}")
        steps = 0
        game = Game(n)
        losses = [0] * n

        while not game.is_over():
            steps += 1
            player_id = game.turn

            trainer = trainers[player_id]
            reward, action = game.step(trainer.policy_net)
            Q = action.score
            with torch.no_grad():
                next_action_score = 0
                if not game.is_over():
                    next_action = trainer.target_net.choose(game)
                    next_action_score = next_action.score
            # Compute the expected Q values
            Q_prime = (next_action_score * gamma) + torch.tensor([reward])
            # print(Q, next_action_score, reward)
            criterion = nn.SmoothL1Loss()
            loss = criterion(Q.unsqueeze(-1), Q_prime)

            losses[player_id] += loss

            if steps % batch_size == 0:
                optimize()

                losses = [0] * n

            if steps == 100:
                break

        if steps % batch_size != 0:
            optimize()
        print(game.score, steps)


if __name__ == "__main__":
    GAMES = 1000
    BATCH_SIZE = 4
    GAMMA = .99
    TAU = .005
    LR = 1e-2
    train(GAMES, BATCH_SIZE, GAMMA, TAU, LR)
