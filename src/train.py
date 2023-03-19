import sys
from copy import deepcopy
from datetime import datetime, date

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
    policies = [model for _ in range(n)]

    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    criterion = nn.SmoothL1Loss()

    # Stats
    avg_reward = 0
    all_asks = [0] * 4
    all_declares = [0] * 4
    our_guy_reward = 0
    our_guy_turns = 0

    for g in tqdm(range(games)):
        # print(f"Game {g}")
        steps = 0
        game = Game(n)

        team_list = []
        declare_team_list = []
        model.ask_history = []
        model.declare_history = []
        while not game.is_over():
            optimizer.zero_grad()
            steps += 1
            i = game.turn
            team = game.players[i].team
            reward_dict, action, declare_actions = game.step(policies)
            team_list.append(team)

            if action.success:
                true_reward = torch.tensor(SUCCEEDS)
            else:
                true_reward = torch.tensor(FAILS)

            loss = criterion(true_reward, action.score)
            writer.add_scalar("Step Ask Loss", loss, (g + 1) * (steps + 1))

            if len(declare_actions) > 0:
                true_reward = torch.stack([torch.tensor(GOOD_DECLARE) if action.success else torch.tensor(BAD_DECLARE) for action in declare_actions if action.success is not None])
                declare_scores = torch.stack([action.score for action in declare_actions if action.success is not None])
                declare_loss = criterion(true_reward, declare_scores)
                writer.add_scalar("Step declare Loss", declare_loss, (g + 1) * (steps + 1))
                loss += declare_loss
                declare_team_list.extend([game.players[action.player].team for action in declare_actions])

            loss.backward()
            optimizer.step()

            if steps == 1000:
                break

        optimizer.zero_grad()
        # Reward stats
        our_guy_reward_per_turn = our_guy_reward / (our_guy_turns + 1e-7)
        average_reward_per_turn = game.cumulative_reward / (steps + 1e-7)

        # Update overall statistics
        all_asks[0] = game.positive_asks[0]
        all_asks[1] = game.negative_asks[0]
        all_asks[2] = sum(game.positive_asks)
        all_asks[3] = sum(game.negative_asks)

        all_declares[0] = game.positive_declares[0]
        all_declares[1] = game.negative_declares[0]
        all_declares[2] = sum(game.positive_declares)
        all_declares[3] = sum(game.negative_declares)

        avg_reward_comparison = our_guy_reward_per_turn - average_reward_per_turn

        game_scores = torch.tensor(
            [WIN_GAME if (team == 0 and game.score > 0) or (team == 1 and game.score < 0) else LOSE_GAME for
             team in team_list]).unsqueeze(-1)

        game_output = model(torch.stack(model.ask_history, dim=0), type="ask")
        loss = criterion(game_scores, game_output)

        if len(model.declare_history) > 0:
            game_scores = torch.tensor(
                [WIN_GAME if (team == 0 and game.score > 0) or (team == 1 and game.score < 0) else LOSE_GAME for
                 team in declare_team_list]).unsqueeze(-1)
            game_output = model(torch.stack(model.declare_history, dim=0), type="declare")
            loss += criterion(game_scores, game_output)

        loss.backward()
        optimizer.step()

        # print(f"Ending game score: {game.score}")
        # print(f"Total positive asks: {game.positive_asks}, total negative asks: {game.negative_asks}")
        # print(f"Total positive declares: {game.positive_declares}, total negative declares: {game.negative_declares}")
        # Log the loss and other metrics every 'log_interval' iterations
        log_interval = 1
        if g % log_interval == (log_interval - 1):
            # Compute the average loss over the last 'log_interval' iterations
            # average_loss = running_loss / log_interval

            # Log the loss to TensorBoard
            writer.add_scalar("Train Game Loss", loss, (g + 1))
            writer.add_scalar("Train Steps", steps, (g + 1))
            writer.add_scalar("Train Game Score", game.score, (g + 1))

            writer.add_scalar("Train Declares/Agent + Rate", all_declares[0] / (all_declares[0] + all_declares[1] + 1e-7),
                              (g + 1))
            writer.add_scalar("Train Declares/Everyone + Rate", all_declares[2] / (all_declares[2] + all_declares[3] + 1e-7),
                              ((g + 1)))
            writer.add_scalar("Train Asks/Agent + Rate", all_asks[0] / (all_asks[0] + all_asks[1]), (g + 1))
            writer.add_scalar("Train Asks/Everyone + Rate", all_asks[2] / (all_asks[2] + all_asks[3]), (g + 1))

            # writer.add_scalar("Declares/Agent +", all_declares[0], (g + 1))
            # writer.add_scalar("Declares/Agent -", all_declares[1], (g + 1))
            # writer.add_scalar("Declares/Others +", all_declares[2], (g + 1))
            # writer.add_scalar("Declares/Others -", all_declares[3], (g + 1))
            # writer.add_scalar("Asks/Agent +", all_asks[0], (g + 1))
            # writer.add_scalar("Asks/Agent -", all_asks[1], (g + 1))
            # writer.add_scalar("Asks/Others +", all_asks[2], (g + 1))
            # writer.add_scalar("Asks/Others -", all_asks[3], (g + 1))
        random_vs_random(deepcopy(model).eval(), 1, writer, g)

    torch.save(model, 'model.pt')


def random_vs_random(model, games: int, writer: SummaryWriter, g):
    n = 6
    policies = [model for _ in range(n // 2)] + [MoveEval() for _ in range(n // 2)]
    our_guy_reward = 0
    our_guy_turns = 0

    # Stats
    avg_reward = 0
    all_asks = [0] * 4
    all_declares = [0] * 4
    our_guy_reward = 0
    our_guy_turns = 0


    # print(f"Game {g}")
    steps = 0
    game = Game(n)

    while not game.is_over():
        player_id = game.turn
        steps += 1

        reward, action, _ = game.step(policies)

        if player_id == 0:
            our_guy_reward += reward[player_id]
            our_guy_turns += 1

        if steps == 1000:
            break

    print(f"Ending game score: {game.score}")
    print(f"Average score per turn: {game.cumulative_reward / steps}")
    print(f"Total positive asks: {game.positive_asks}, total negative asks: {game.negative_asks}")

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

    # Log the loss and other metrics every 'log_interval' iterations
    log_interval = 1
    if g % log_interval == (log_interval - 1):
        writer.add_scalar("RandoEval Game Score", game.score, (g + 1))

        writer.add_scalar("RandoEval Declares/Agent + Rate", all_declares[0] / (all_declares[0] + all_declares[1] + 1e-7),
                            (g + 1))
        writer.add_scalar("RandoEval Declares/Everyone + Rate", all_declares[2] / (all_declares[2] + all_declares[3] + 1e-7),
                            ((g + 1)))
        writer.add_scalar("RandoEval Asks/Agent + Rate", all_asks[0] / (all_asks[0] + all_asks[1] + 1e-7), (g + 1))
        writer.add_scalar("RandoEval Asks/Everyone + Rate", all_asks[2] / (all_asks[2] + all_asks[3] + 1e-7), (g + 1))


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
    # random_vs_random(GAMES, WRITER)
    WRITER.close()


if __name__ == "__main__":
    main()
