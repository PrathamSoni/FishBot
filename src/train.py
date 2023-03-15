import copy

from game import Game
from models import RecurrentTrainer, RecurrentTrainer2
from policy import RandomPolicy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from datetime import datetime, date


def train(games, batch_size, gamma, tau, lr):
    n = 6
    # Select trainer here
    trainers = [RecurrentTrainer2(i) for i in range(n)]
    # trainers = [RecurrentTrainer(i, tau) for i in range(n)]
    optimizers = [optim.AdamW(trainer.policy_net.parameters(), lr=lr, amsgrad=True) for trainer in trainers]

    for g in range(games):
        print(f"Game {g}")
        steps = 0
        game = Game(n)
        losses = [0] * n
        rewards = [0] * n
        last_state = [None] * n
        while not game.is_over():
            steps += 1
            player_id = game.turn
            trainer = trainers[player_id]
            optimizer = optimizers[player_id]

            if (last := last_state[player_id]) is not None:
                optimizer.zero_grad()
                Q, reward = last

                with torch.no_grad():
                    next_action_score = 0
                    if not game.is_over():
                        next_action = trainer.target_net.choose(game)
                        next_action_score = next_action.score

                # Compute the expected Q values
                Q_prime = (next_action_score * gamma) + torch.tensor([reward])
                # print(Q, next_action_score, reward, Q_prime)
                criterion = nn.SmoothL1Loss()
                loss = criterion(Q.unsqueeze(-1), Q_prime)
                loss.backward()
                torch.nn.utils.clip_grad_value_(trainer.policy_net.parameters(), 100)
                optimizer.step()

                trainer.update_target()

            reward, action = game.step(trainer.policy_net)
            rewards[player_id] += int(reward > 0)
            last_state[player_id] = (action.score, reward)

            if steps == 1000:
                break

        print(game.score, steps, rewards)
        print(f"Ending game score: {game.score}")
        print(f"Average score per turn: {game.cumulative_reward / steps}")
        print(f"Total positive asks: {game.positive_asks}, total negative asks: {game.negative_asks}")


def levels_train(levels, games, batch_size, gamma, tau, lr):
    time_string = "" + date.today().strftime("%d-%m-%Y") + "_" + datetime.now().strftime("%H-%M-%S")
    n = 6
    our_guy = RecurrentTrainer(0, tau)
    other_guy = RandomPolicy()
    optimizer = optim.AdamW(our_guy.policy_net.parameters(), lr=lr, amsgrad=True)
    last = None

    avg_reward = 0
    for l in range(levels):
        for g in range(games):
            print(f"Game {g}")
            steps = 0
            game = Game(n)
            loss = 0

            our_guy_reward = 0
            our_guy_turns = 0

            while not game.is_over():
                player_id = game.turn
                steps += 1
                if player_id == 0:
                    if last is not None:
                        optimizer.zero_grad()
                        Q, reward = last

                        with torch.no_grad():
                            next_action_score = 0
                            if not game.is_over():
                                next_action = our_guy.target_net.choose(game)
                                next_action_score = next_action.score

                        # Compute the expected Q values
                        Q_prime = (next_action_score * gamma) + torch.tensor([reward])
                        # print(Q, next_action_score, reward, Q_prime)
                        criterion = nn.SmoothL1Loss()
                        loss = criterion(Q.unsqueeze(-1), Q_prime)
                        loss.backward()
                        torch.nn.utils.clip_grad_value_(our_guy.policy_net.parameters(), 100)
                        optimizer.step()

                        our_guy.update_target()

                    reward, action = game.step(our_guy.policy_net)
                    last = (action.score, reward)
                    our_guy_reward += int(reward)
                    our_guy_turns += 1

                else:
                    game.step(other_guy)

                if steps == 1000:
                    break

            # print(game.score, steps)
            # print(f"Ending game score: {game.score}")
            print(f"Our guy's reward per turn: {our_guy_reward / our_guy_turns}")
            print(f"Our guy's positive asks: {game.positive_asks[0]}, our guy's negative asks: {game.negative_asks[0]}")
            
            # print(f"Average reward per turn: {game.cumulative_reward / steps}")
            # print(f"Total positive asks: {sum(game.positive_asks)}, total negative asks: {sum(game.negative_asks)}")
            avg_reward_comparison = our_guy_reward / (our_guy_turns + 1e-7) - game.cumulative_reward / (steps + 1e-7)
            avg_reward += avg_reward_comparison

            print(f"Our guy's reward/turn vs other guy's reward/turn (positive = better): {avg_reward_comparison}")
            print(f"Our guy's correct ask percentage vs other guy's ask percentage (positive = better): {game.positive_asks[0] / (game.positive_asks[0] + game.negative_asks[0] + 1e-7) - sum(game.positive_asks) / (sum(game.positive_asks) + sum(game.negative_asks) + 1e-7)}")
        other_guy = deepcopy(our_guy.policy_net)
        torch.save(our_guy.policy_net, f"{time_string}_level{l}.pt")

    print(f"Final average reward/turn vs other guy (positive = better): {avg_reward / (levels * games)}, over {levels * games} games")


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    LEVELS = 3
    GAMES = 1000
    BATCH_SIZE = 4
    GAMMA = .99
    TAU = .005
    LR = 1e-2
    levels_train(LEVELS, GAMES, BATCH_SIZE, GAMMA, TAU, LR)
    # GAMES = 100
    # BATCH_SIZE = 1
    # GAMMA = .99
    # TAU = .05
    # LR = 1e-1
    # train(GAMES, BATCH_SIZE, GAMMA, TAU, LR)
