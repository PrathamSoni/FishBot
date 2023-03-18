import sys
from copy import deepcopy
from datetime import datetime, date

import torch
import torch.nn as nn
import torch.optim as optim

from expert_model import RecurrentTrainer2
from game import Game
from policy import RandomPolicy
from move_eval import MoveEval
from utils import *

from torch.utils.tensorboard import SummaryWriter


def train(games, lr, outfile, writer):
    n = 6
    # Select trainer here
    model = MoveEval()
    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    criterion = nn.SmoothL1Loss()

    avg_reward = 0
    all_asks = [0] * 4
    all_declares = [0] * 4
    our_guy_reward = 0
    our_guy_turns = 0
    game_loss = 0

    for g in range(games):
        print(f"Game {g}")
        steps = 0
        game = Game(n)
        game.turn = 0

        team_list = []
        model.history = []
        while not game.is_over():
            steps += 1
            i = game.turn
            team = game.players[i].team
            reward_dict, action = game.step(model)
            team_list.append(team)

            if action.success:
                true_reward = torch.tensor(SUCCEEDS)
            else:
                true_reward = torch.tensor(FAILS)

            loss = criterion(true_reward, action.score)

            writer.add_scalar("Step Loss", loss, (g + 1) * (steps + 1))

            loss.backward()
            optimizer.step()

            if steps == 200:
                break

        # Reward stats
        our_guy_reward_per_turn = our_guy_reward / (our_guy_turns + 1e-7)
        average_reward_per_turn = game.cumulative_reward / (steps + 1e-7)

        # Update overall statistics
        all_asks[0] += game.positive_asks[0]
        all_asks[1] += game.negative_asks[0]
        all_asks[2] += sum(game.positive_asks)
        all_asks[3] += sum(game.negative_asks)

        all_declares[0] += game.positive_declares[0]
        all_declares[1] += game.negative_declares[0]
        all_declares[2] += sum(game.positive_declares)
        all_declares[3] += sum(game.negative_declares)

        avg_reward_comparison = our_guy_reward_per_turn - average_reward_per_turn
        avg_reward += avg_reward_comparison

        game_scores = torch.tensor([WIN_GAME if (team == 0 and game.score > 0) or (team == 1 and game.score < 0) else LOSE_GAME for
                       team in team_list]).unsqueeze(-1)

        game_output = model(torch.stack(model.history, dim=0))
        loss = criterion(game_scores, game_output)
        loss.backward()
        optimizer.step()

        print(f"Ending game score: {game.score}")
        print(f"Average score per turn: {game.cumulative_reward / steps}")
        print(f"Total positive asks: {game.positive_asks}, total negative asks: {game.negative_asks}")

        # Log the loss and other metrics every 'log_interval' iterations
        log_interval = 1
        if g % log_interval == (log_interval - 1):
            # Compute the average loss over the last 'log_interval' iterations
            # average_loss = running_loss / log_interval

            # Log the loss to TensorBoard
            writer.add_scalar("Loss", loss, (g + 1))
            writer.add_scalar("Game Loss", game_loss, (g + 1))
            writer.add_scalar("Game Score", game.score, (g + 1))

            writer.add_scalar("Declares/Agent + Rate", all_declares[0] / (all_declares[0] + all_declares[1] + 1e-7),
                              (g + 1))
            writer.add_scalar("Declares/Everyone + Rate", all_declares[2] / (all_declares[2] + all_declares[3] + 1e-7),
                              ((g + 1)))
            writer.add_scalar("Asks/Agent + Rate", all_asks[0] / (all_asks[0] + all_asks[1]), (g + 1))
            writer.add_scalar("Asks/Everyone + Rate", all_asks[2] / (all_asks[2] + all_asks[3]), (g + 1))

            # writer.add_scalar("Declares/Agent +", all_declares[0], (g + 1))
            # writer.add_scalar("Declares/Agent -", all_declares[1], (g + 1))
            # writer.add_scalar("Declares/Others +", all_declares[2], (g + 1))
            # writer.add_scalar("Declares/Others -", all_declares[3], (g + 1))
            # writer.add_scalar("Asks/Agent +", all_asks[0], (g + 1))
            # writer.add_scalar("Asks/Agent -", all_asks[1], (g + 1))
            # writer.add_scalar("Asks/Others +", all_asks[2], (g + 1))
            # writer.add_scalar("Asks/Others -", all_asks[3], (g + 1))

def random_vs_random(games: int):
    n = 6
    policies = [RandomPolicy() for _ in range(n)]
    our_guy_reward = 0
    our_guy_turns = 0

    for g in range(games):
        print(f"Game {g}")
        steps = 0
        game = Game(n)
        game.turn = 0

        while not game.is_over():
            player_id = game.turn
            steps += 1

            reward, action = game.step(policies[player_id])

            if player_id == 0:
                our_guy_reward += reward[player_id]
                our_guy_turns += 1

            if steps == 200:
                break


        print(f"Ending game score: {game.score}")
        print(f"Average score per turn: {game.cumulative_reward / steps}")
        print(f"Total positive asks: {game.positive_asks}, total negative asks: {game.negative_asks}")

def levels_train(levels, games, gamma, tau, lr, outfile, writer):
    time_string = "" + date.today().strftime("%d-%m-%Y") + "_" + datetime.now().strftime("%H-%M-%S")
    n = 6
    # our_guy = RecurrentTrainer(0, tau)
    our_guy = RecurrentTrainer2(0, tau)

    other_guy = RandomPolicy()
    optimizer = optim.AdamW(our_guy.policy_net.parameters(), lr=lr, amsgrad=True)
    last = None

    avg_reward = 0
    all_asks = [0] * 4
    all_declares = [0] * 4
    for level in range(levels):
        print(f"Level {level}")
        for g in range(games):
            # print(f"Game {g}")
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
                        torch.nn.utils.clip_grad_value_(our_guy.policy_net.parameters(), 1)
                        optimizer.step()

                        our_guy.update_target()

                    reward, action = game.step(our_guy.policy_net)
                    last = (action.score, reward)
                    our_guy_reward += int(reward)
                    our_guy_turns += 1

                else:
                    setattr(other_guy, "i", torch.tensor([game.turn]))
                    game.step(other_guy)

                if steps == 100:
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

        other_guy = deepcopy(our_guy.policy_net)
        # torch.save(our_guy.policy_net, f"{time_string}_level{l}.pt")

        # Log the loss and other metrics every 'log_interval' iterations
        log_interval = 1
        if g % log_interval == (log_interval - 1):
            # Compute the average loss over the last 'log_interval' iterations
            # average_loss = running_loss / log_interval

            # Log the loss to TensorBoard
            writer.add_scalar("Declares/Agent +", all_declares[0], (level + 1) * (g + 1))
            writer.add_scalar("Declares/Agent -", all_declares[1], (level + 1) * (g + 1))
            writer.add_scalar("Declares/Others +", all_declares[2], (level + 1) * (g + 1))
            writer.add_scalar("Declares/Others -", all_declares[3], (level + 1) * (g + 1))
            writer.add_scalar("Declares/Agent + Rate", all_declares[0] / (all_declares[0] + all_declares[1]),
                              (level + 1) * (g + 1))
            writer.add_scalar("Declares/Others + Rate", all_declares[2] / (all_declares[2] + all_declares[3]),
                              (level + 1) * (g + 1))
            print(all_declares, all_asks)
            writer.add_scalar("Asks/Agent +", all_asks[0], (level + 1) * (g + 1))
            writer.add_scalar("Asks/Agent -", all_asks[1], (level + 1) * (g + 1))
            writer.add_scalar("Asks/Others +", all_asks[2], (level + 1) * (g + 1))
            writer.add_scalar("Asks/Others -", all_asks[3], (level + 1) * (g + 1))
            writer.add_scalar("Asks/Agent + Rate", all_asks[0] / (all_asks[0] + all_asks[1]), (level + 1) * (g + 1))
            writer.add_scalar("Asks/Others + Rate", all_asks[2] / (all_asks[2] + all_asks[3]), (level + 1) * (g + 1))

            # Reset the running loss
            # running_loss = 0.0

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
    # torch.autograd.set_detect_anomaly(True)
    if len(sys.argv) != 3:
        raise Exception("usage: python train.py <outfile>.txt num_games")
    # Set with params
    OUTFILE = sys.argv[1]
    GAMES = int(sys.argv[2])

    LR = 1e-4

    WRITER = SummaryWriter(f"runs/{OUTFILE}")

    train(GAMES, LR, OUTFILE, WRITER)
    # train(GAMES, LR)
    # random_vs_random(GAMES)
    WRITER.close()


if __name__ == "__main__":
    main()
