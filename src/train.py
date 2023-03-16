import sys
from copy import deepcopy
from datetime import datetime, date

import torch
import torch.nn as nn
import torch.optim as optim

from expert_model import RecurrentTrainer2
from game import Game
from policy import RandomPolicy
from models import RecurrentTrainer

'''
# Currently not in use. This is the old training loop.
def train(games, batch_size, gamma, tau, lr):
    n = 6
    # Select trainer here
    # trainers = [RecurrentTrainer2(i) for i in range(n)]
    trainers = [RecurrentTrainer(i, tau) for i in range(n)]
    optimizers = [optim.AdamW(trainer.policy_net.parameters(), lr=lr, amsgrad=True) for trainer in trainers]

    for g in range(games):
        print(f"Game {g}")
        steps = 0
        game = Game(n)
        game.turn = 0
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

            if steps == 100:
                break

        print(game.score, steps, rewards)
        print(f"Ending game score: {game.score}")
        print(f"Average score per turn: {game.cumulative_reward / steps}")
        print(f"Total positive asks: {game.positive_asks}, total negative asks: {game.negative_asks}")
'''


def levels_train(levels, games, batch_size, gamma, tau, lr, outfile):
    time_string = "" + date.today().strftime("%d-%m-%Y") + "_" + datetime.now().strftime("%H-%M-%S")
    n = 6
    # our_guy = RecurrentTrainer(0, tau)
    our_guy = RecurrentTrainer(0)

    other_guy = RandomPolicy()
    optimizer = optim.AdamW(our_guy.policy_net.parameters(), lr=lr, amsgrad=True)
    last = None

    avg_reward = 0
    all_asks = [0] * 4
    all_declares = [0] * 4
    for level in range(levels):
        print(f"Level {level}")
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
                    setattr(other_guy, "i", torch.tensor([game.turn]))
                    game.step(other_guy)

                if steps == 10000:
                    break

            # Reward stats
            our_guy_reward_per_turn = our_guy_reward / (our_guy_turns + 1e-7)
            average_reward_per_turn = game.cumulative_reward / (steps + 1e-7)

            # Ask stats
            our_guy_positive_asks = game.positive_asks[0]
            our_guy_negative_asks = game.negative_asks[0]
            other_guy_positive_asks = sum(game.positive_asks) - game.positive_asks[0]
            other_guy_negative_asks = sum(game.negative_asks) - game.negative_asks[0]

            # Declare stats
            our_guy_positive_declares = game.positive_declares[0]
            our_guy_negative_declares = game.negative_declares[0]
            other_guy_positive_declares = sum(game.positive_declares) - game.positive_declares[0]
            other_guy_negative_declares = sum(game.negative_declares) - game.negative_declares[0]

            # Update overall statistics
            all_asks[0] += our_guy_positive_asks
            all_asks[1] += our_guy_negative_asks
            all_asks[2] += other_guy_positive_asks
            all_asks[3] += other_guy_negative_asks

            all_declares[0] += our_guy_positive_declares
            all_declares[1] += our_guy_negative_declares
            all_declares[2] += other_guy_positive_declares
            all_declares[3] += other_guy_negative_declares

            avg_reward_comparison = our_guy_reward_per_turn - average_reward_per_turn
            avg_reward += avg_reward_comparison

            # print(f"Our guy's reward per turn: {our_guy_reward_per_turn}") print(f"Our guy's positive asks: {
            # our_guy_positive_asks}, our guy's negative asks: {our_guy_negative_asks}") print(f"Our guy's positive
            # declares: {our_guy_positive_declares}, our guy's negative declares: {our_guy_negative_declares}")
            # print(f"Other guy's positive declares: {other_guy_positive_declares}, other guy's negative declares: {
            # other_guy_negative_declares}") print(f"Average reward per turn: {average_reward_per_turn}") print(
            # f"Other guy's positive asks: {other_guy_positive_asks}, other guy's negative asks: {
            # other_guy_negative_asks}")

            # print(f"Our guy's reward/turn vs other guy's reward/turn (positive = better): {avg_reward_comparison}")
            # print(f"Our guy's correct ask percentage vs other guy's ask percentage (positive = better): \ {
            # game.positive_asks[0] / (game.positive_asks[0] + game.negative_asks[0] + 1e-7) - sum(
            # game.positive_asks) / (sum(game.positive_asks) + sum(game.negative_asks) + 1e-7)}")

        other_guy = deepcopy(our_guy.policy_net)
        # torch.save(our_guy.policy_net, f"{time_string}_level{l}.pt")

        print(
            f"Average reward/turn vs other guy (positive = better): {avg_reward / ((level + 1) * games)}, over {(level + 1) * games} games\n")
        with open(f"{outfile}_{levels}_levels_{games}_games.txt", 'a') as f:
            f.write(f"Level {level} STATISTICS\n")
            f.write(
                f"Our guy total positive declares: {all_declares[0]}, our guy total negative declares: {all_declares[1]}\n")
            f.write(
                f"Other guy total positive declares: {all_declares[2]}, Other guy total negative declares: {all_declares[3]}\n")
            f.write(f"Our guy total positive asks: {all_asks[0]}, our guy total negative asks: {all_asks[1]}\n")
            f.write(f"Other guy total negative asks: {all_asks[2]}, Other guy total negative asks: {all_asks[3]}\n")
    print(
        f"Final average reward/turn vs other guy (positive = better): {avg_reward / (levels * games)}, over {levels * games} games\n")

    with open(f"{outfile}_{levels}_levels_{games}_games.txt", 'a') as f:
        f.write(f"FINAL STATISTICS\n")
        f.write(
            f"Our guy total positive declares: {all_declares[0]}, our guy total negative declares: {all_declares[1]}\n")
        f.write(
            f"Other guy total positive declares: {all_declares[2]}, Other guy total negative declares: {all_declares[3]}\n")
        f.write(f"Our guy total positive asks: {all_asks[0]}, our guy total negative asks: {all_asks[1]}\n")
        f.write(f"Other guy total negative asks: {all_asks[2]}, Other guy total negative asks: {all_asks[3]}\n")
        f.write(
            f"Final average reward/turn vs other guy (positive = better): {avg_reward / (levels * games)}, over {levels * games} games\n")


def main():
    if len(sys.argv) != 4:
        raise Exception("usage: python train.py <outfile>.txt num_levels num_games_per_level")
    # Set with params
    OUTFILE = sys.argv[1]
    LEVELS = int(sys.argv[2])
    GAMES = int(sys.argv[3])

    BATCH_SIZE = 4
    GAMMA = .99
    TAU = .005
    LR = 1e-2
    levels_train(LEVELS, GAMES, BATCH_SIZE, GAMMA, TAU, LR, OUTFILE)


if __name__ == "__main__":
    main()
