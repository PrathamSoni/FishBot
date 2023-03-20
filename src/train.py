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


def train(games, lr, writer):
    n = 6
    # Select trainer here
    model = MoveEval()
    policies = [model for i in range(n)]
    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    criterion = nn.SmoothL1Loss()

    # Stats
    all_asks = [0] * 4
    all_declares = [0] * 4

    logging_steps = 0
    logging_score = 0
    logging_game_loss = 0

    log_interval = 5

    for g in tqdm(range(games)):
        # print(f"Game {g}")
        steps = 0
        logging_ask_loss = 0
        logging_declare_loss = 0
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

            logging_ask_loss += loss / log_interval
            if g % log_interval == (log_interval - 1):
                writer.add_scalar("Step Ask Loss", logging_ask_loss, (g + 1) * (steps + 1))
                loging_loss = 0

            if len(declare_actions) > 0:
                true_reward = torch.stack(
                    [torch.tensor(GOOD_DECLARE) if action.success else torch.tensor(BAD_DECLARE) for action in
                     declare_actions if action.success is not None])
                declare_scores = torch.stack([action.score for action in declare_actions if action.success is not None])
                declare_loss = criterion(true_reward, declare_scores)

                logging_declare_loss += declare_loss / log_interval
                if g % log_interval == (log_interval - 1):
                    writer.add_scalar("Step Declare Loss", logging_declare_loss, (g + 1) * (steps + 1))

                loss += declare_loss
                declare_team_list.extend([game.players[action.player].team for action in declare_actions])

            loss.backward()
            optimizer.step()

            if steps == 1000:
                break

        optimizer.zero_grad()
        # Reward stats

        # Update overall statistics
        all_asks[0] += (game.positive_asks[0] + game.positive_asks[1] + game.positive_asks[2]) / log_interval
        all_asks[1] += (game.negative_asks[0] + game.negative_asks[1] + game.negative_asks[2]) / log_interval
        all_asks[2] += (game.positive_asks[3] + game.positive_asks[4] + game.positive_asks[5]) / log_interval
        all_asks[3] += (game.negative_asks[3] + game.negative_asks[4] + game.negative_asks[5]) / log_interval

        all_declares[0] += (game.positive_declares[0] + game.positive_declares[1] + game.positive_declares[2]) / log_interval
        all_declares[1] += (game.negative_declares[0] + game.negative_declares[1] + game.negative_declares[2]) / log_interval
        all_declares[2] += (game.positive_declares[3] + game.positive_declares[4] + game.positive_declares[5]) / log_interval
        all_declares[3] += (game.negative_declares[3] + game.negative_declares[4] + game.negative_declares[5]) / log_interval

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

        logging_game_loss += loss / log_interval
        logging_steps += steps / log_interval
        logging_score += game.score / log_interval

        loss.backward()
        optimizer.step()

        # Log the loss and other metrics every 'log_interval' iterations
        if g % log_interval == (log_interval - 1):
            # Compute the average loss over the last 'log_interval' iterations
            # average_loss = running_loss / log_interval

            # Log the loss to TensorBoard
            writer.add_scalar("Game Loss/Train", logging_game_loss, (g + 1))
            writer.add_scalar("Steps/Train", logging_steps, (g + 1))
            writer.add_scalar("Game Score/Train", logging_score, (g + 1))

            writer.add_scalar("Declares/Train/Team 1 Success Rate", all_declares[0] / (all_declares[0] + all_declares[1] + 1e-7),
                              (g + 1))
            writer.add_scalar("Declares/Train/Team 2 Success Rate", all_declares[2] / (all_declares[2] + all_declares[3] + 1e-7),
                              (g + 1))
            writer.add_scalar("Asks/Train/Team 1 Success Rate", all_asks[0] / (all_asks[0] + all_asks[1]), (g + 1))
            writer.add_scalar("Asks/Train/Team 2 Success Rate", all_asks[2] / (all_asks[2] + all_asks[3]), (g + 1))

            all_asks = [0] * 4
            all_declares = [0] * 4

            logging_steps = 0
            logging_game_loss = 0
            logging_score = 0
        # deepcopy(model).eval()
        eval(model, RandomPolicy(), g, writer)

    torch.save(model, 'model.pt')


def eval(model, eval_model, g: int, writer: SummaryWriter):
    n = 6
    policies = [model for _ in range(n // 2)] + [eval_model for _ in range(n // 2)]
    our_guy_reward = 0
    our_guy_turns = 0

    # Stats
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

        reward, _, _ = game.step(policies)

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
    all_asks[0] = (game.positive_asks[0] + game.positive_asks[1] + game.positive_asks[2])
    all_asks[1] = (game.negative_asks[0] + game.negative_asks[1] + game.negative_asks[2])
    all_asks[2] = (game.positive_asks[3] + game.positive_asks[4] + game.positive_asks[5])
    all_asks[3] = (game.negative_asks[3] + game.negative_asks[4] + game.negative_asks[5])

    all_declares[0] = (game.positive_declares[0] + game.positive_declares[1] + game.positive_declares[2])
    all_declares[1] = (game.negative_declares[0] + game.negative_declares[1] + game.negative_declares[2])
    all_declares[2] = (game.positive_declares[3] + game.positive_declares[4] + game.positive_declares[5])
    all_declares[3] = (game.negative_declares[3] + game.negative_declares[4] + game.negative_declares[5])


    # Log the loss and other metrics every 'log_interval' iterations
    log_interval = 1
    if g % log_interval == (log_interval - 1):
        writer.add_scalar("Game Score/Eval", game.score, (g + 1))

        writer.add_scalar("Declares/Eval/Team 1 Success Rate", all_declares[0] / (all_declares[0] + all_declares[1] + 1e-7),
                          (g + 1))
        writer.add_scalar("Declares/Eval/Team 2 Success Rate", all_declares[2] / (all_declares[2] + all_declares[3] + 1e-7),
                          ((g + 1)))
        writer.add_scalar("Asks/Eval/Team 1 Success Rate", all_asks[0] / (all_asks[0] + all_asks[1] + 1e-7), (g + 1))
        writer.add_scalar("Asks/Eval/Team 2 Success Rate", all_asks[2] / (all_asks[2] + all_asks[3] + 1e-7), (g + 1))


def main():
    # torch.autograd.set_detect_anomaly(True)
    if len(sys.argv) != 3:
        raise Exception("usage: python train.py <outfile>.txt num_games")
    # Set with params
    OUTFILE = sys.argv[1]
    GAMES = int(sys.argv[2])

    LR = 1e-4

    WRITER = SummaryWriter(f"runs/{OUTFILE}")

    train(GAMES, LR, WRITER)
    # random_vs_random(GAMES, WRITER)
    WRITER.close()


if __name__ == "__main__":
    main()
