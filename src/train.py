from game import Game
from models import RecurrentTrainer
from policy import RandomPolicy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from datetime import datetime, date

def train(games, batch_size, gamma, tau, lr):
    n = 6
    trainers = [RecurrentTrainer(i, tau) for i in range(n)]
    optimizers = [optim.AdamW(trainer.policy_net.parameters(), lr=lr, amsgrad=True) for trainer in trainers]

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


def levels_train(levels, games, batch_size, gamma, tau, lr):
    time_string = "" + date.today().strftime("%d-%m-%Y") + "_" + datetime.now().strftime("%H-%M-%S")
    n = 6
    our_guy = RecurrentTrainer(0, tau)
    other_guy = RandomPolicy()
    optimizer = optim.AdamW(our_guy.policy_net.parameters(), lr=lr, amsgrad=True)

    def optimize():
        # print(losses)
        if type(loss) != int:
            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_value_(our_guy.policy_net.parameters(), 100)
            optimizer.step()

            our_guy.update_target()

    for l in range(levels):
        for g in range(games):
            print(f"Game {g}")
            steps = 0
            game = Game(n)
            loss = 0

            while not game.is_over():
                player_id = game.turn
                steps += 1
                if player_id == 0:
                    reward, action = game.step(our_guy.policy_net)
                    Q = action.score
                    with torch.no_grad():
                        next_action_score = 0
                        if not game.is_over():
                            next_action = our_guy.target_net.choose(game)
                            next_action_score = next_action.score
                    # Compute the expected Q values
                    Q_prime = (next_action_score * gamma) + torch.tensor([reward])
                    # print(Q, next_action_score, reward)
                    criterion = nn.SmoothL1Loss()
                    loss += criterion(Q.unsqueeze(-1), Q_prime)

                    if steps % batch_size == 0:
                        optimize()
                        loss = 0
                else:
                    game.step(other_guy)
                if steps == 100:
                    break
            if steps % batch_size != 0:
                optimize()
            print(game.score, steps)
        other_guy = deepcopy(our_guy.policy_net)
        torch.save(our_guy.policy_net, f"{time_string}_level{l}.pt")


if __name__ == "__main__":
    LEVELS = 10
    GAMES = 1000
    BATCH_SIZE = 4
    GAMMA = .99
    TAU = .005
    LR = 1e-2
    levels_train(LEVELS, GAMES, BATCH_SIZE, GAMMA, TAU, LR)
