# predict.py

import numpy as np
import random
import statistics
from typing import List, Dict, Tuple
from scipy import stats  # For confidence intervals


# Player class representing a tennis player
class Player:
    def __init__(
        self,
        name: str,
        serve_percentage: float,
        break_point_conversion: float,
        first_serve_won: float,
        second_serve_won: float,
        sets_won: int = 0,
        games_won: int = 0,
        points_won: int = 0,
    ):
        self.name = name
        self.serve_percentage = serve_percentage  # Probability of serve being in (0 to 1)
        self.break_point_conversion = break_point_conversion
        self.first_serve_won = first_serve_won  # Probability of winning the point on first serve (0 to 1)
        self.second_serve_won = second_serve_won  # Probability of winning the point on second serve (0 to 1)
        self.sets_won = sets_won
        self.games_won = games_won
        self.points_won = points_won


# Match class simulating a tennis match between two players
class Match:
    def __init__(self, player1: Player, player2: Player):
        self.player1 = player1
        self.player2 = player2

    def simulate_point(self, server: Player, receiver: Player) -> Player:
        serve_in = random.random() < server.serve_percentage
        if serve_in:
            # First serve
            point_won = random.random() < server.first_serve_won
        else:
            # Second serve
            point_won = random.random() < server.second_serve_won
        return server if point_won else receiver

    def simulate_game(self, server: Player, receiver: Player) -> Player:
        points_server = 0
        points_receiver = 0
        while True:
            point_winner = self.simulate_point(server, receiver)
            if point_winner == server:
                points_server += 1
            else:
                points_receiver += 1

            # Check for game win
            if points_server >= 4 and points_server - points_receiver >= 2:
                return server  # Server wins the game
            if points_receiver >= 4 and points_receiver - points_server >= 2:
                return receiver  # Receiver wins the game

    def simulate_tie_break(self, server: Player, receiver: Player) -> Player:
        points_p1 = 0
        points_p2 = 0
        current_server = server
        current_receiver = receiver
        serve_changes = 0
        total_points = 0

        while True:
            point_winner = self.simulate_point(current_server, current_receiver)
            if point_winner == self.player1:
                points_p1 += 1
            else:
                points_p2 += 1

            total_points += 1
            serve_changes += 1

            # Change server after first point, then every two points
            if (total_points == 1) or (serve_changes == 2):
                current_server, current_receiver = current_receiver, current_server
                serve_changes = 0

            # Check for tie-break win
            if points_p1 >= 7 and points_p1 - points_p2 >= 2:
                return self.player1
            if points_p2 >= 7 and points_p2 - points_p1 >= 2:
                return self.player2

    def simulate_set(self, server: Player, receiver: Player) -> Player:
        games_p1 = 0
        games_p2 = 0
        current_server = server
        current_receiver = receiver

        while True:
            game_winner = self.simulate_game(current_server, current_receiver)
            if game_winner == self.player1:
                games_p1 += 1
            else:
                games_p2 += 1

            # Check for set win
            if games_p1 >= 6 and games_p1 - games_p2 >= 2:
                return game_winner
            if games_p2 >= 6 and games_p2 - games_p1 >= 2:
                return game_winner

            # Handle tie-break at 6-6
            if games_p1 == 6 and games_p2 == 6:
                tie_break_winner = self.simulate_tie_break(current_server, current_receiver)
                return tie_break_winner

            # Alternate server for next game
            current_server, current_receiver = current_receiver, current_server

    def simulate_match(self) -> Player:
        sets_p1 = 0
        sets_p2 = 0

        # Randomly decide who serves first
        if random.random() < 0.5:
            server = self.player1
            receiver = self.player2
        else:
            server = self.player2
            receiver = self.player1

        while sets_p1 < 2 and sets_p2 < 2:
            set_winner = self.simulate_set(server, receiver)
            if set_winner == self.player1:
                sets_p1 += 1
            else:
                sets_p2 += 1

            # Alternate server for next set
            server, receiver = receiver, server

        return self.player1 if sets_p1 >= 2 else self.player2


def compute_average_stats(stats_list: List[Dict], name: str) -> Dict:
    """Compute weighted average statistics from match history."""
    if not stats_list:
        raise ValueError("Stats list cannot be empty")

    weights = np.exp(np.linspace(0, 1, len(stats_list)))
    weights = weights / np.sum(weights)

    avg_stats = {
        'name': name,
        'serve_percentage': np.average(
            [s['serve_percentage'] for s in stats_list], weights=weights
        ),
        'break_point_conversion': np.average(
            [s['break_point_conversion'] for s in stats_list], weights=weights
        ),
        'first_serve_won': np.average(
            [s['first_serve_won'] for s in stats_list], weights=weights
        ),
        'second_serve_won': np.average(
            [s['second_serve_won'] for s in stats_list], weights=weights
        ),
        'sets_won': 0,
        'games_won': 0,
        'points_won': 0
    }
    return avg_stats


def predict_match(
    player1_avg_stats: Dict, player2_avg_stats: Dict, num_simulations: int = 1000
) -> Tuple[str, float, Tuple[float, float]]:
    """Predict match winner and return winner name with win probability and confidence interval."""
    player1_wins = 0
    player2_wins = 0

    for _ in range(num_simulations):
        # Add random variation based on historical performance
        p1_stats = {
            k: random.gauss(v, 0.05 * v) if isinstance(v, float) else v
            for k, v in player1_avg_stats.items()
        }
        p2_stats = {
            k: random.gauss(v, 0.05 * v) if isinstance(v, float) else v
            for k, v in player2_avg_stats.items()
        }

        # Clamp probabilities between 0 and 1
        for game_stats in [p1_stats, p2_stats]:
            for key in [
                'serve_percentage',
                'break_point_conversion',
                'first_serve_won',
                'second_serve_won'
            ]:
                game_stats[key] = max(0, min(1, game_stats[key]))

        # Create Player instances
        player1 = Player(**p1_stats)
        player2 = Player(**p2_stats)

        # Simulate the match
        match = Match(player1, player2)
        winner = match.simulate_match()

        if winner.name == player1_avg_stats['name']:
            player1_wins += 1
        else:
            player2_wins += 1

    # Calculate win rates
    player1_win_rate = player1_wins / num_simulations
    player2_win_rate = player2_wins / num_simulations

    # Confidence intervals using normal approximation
    confidence_interval = stats.norm.interval(
        0.95,
        loc=player1_win_rate,
        scale=np.sqrt((player1_win_rate * (1 - player1_win_rate)) / num_simulations)
    )

    # Predict the winner
    predicted_winner = (
        player1_avg_stats['name'] if player1_win_rate > player2_win_rate else player2_avg_stats['name']
    )

    return predicted_winner, player1_win_rate, confidence_interval


def collect_player_stats(player_number: int) -> List[Dict]:
    """Collect statistics for a player from user input."""
    stats_list = []
    print(f"\nEnter statistics for Player {player_number} (type 'done' to finish entering stats):")
    while True:
        try:
            game_stats = {}
            game_number = len(stats_list) + 1
            print(f"\nGame #{game_number}:")
            serve_percentage = input("Serve Percentage (0 to 100): ").strip()
            if serve_percentage.lower() == 'done':
                break
            game_stats['serve_percentage'] = float(serve_percentage) / 100

            break_point_conversion = input("Break Point Conversion (0 to 100): ").strip()
            if break_point_conversion.lower() == 'done':
                break
            game_stats['break_point_conversion'] = float(break_point_conversion) / 100

            first_serve_won = input("First Serve Won (0 to 100): ").strip()
            if first_serve_won.lower() == 'done':
                break
            game_stats['first_serve_won'] = float(first_serve_won) / 100

            second_serve_won = input("Second Serve Won (0 to 100): ").strip()
            if second_serve_won.lower() == 'done':
                break
            game_stats['second_serve_won'] = float(second_serve_won) / 100

            stats_list.append(game_stats)

            cont = input("Add another game's stats for this player? (y/n): ").strip().lower()
            if cont != 'y':
                break
        except ValueError:
            print("Invalid input. Please enter numerical values between 0 and 100.")

    return stats_list


if __name__ == "__main__":
    # Collect stats for Player 1
    print("Collecting stats for Player 1:")
    player1_name = input("Enter Player 1 Name: ").strip()
    player1_stats_list = collect_player_stats(1)

    # Collect stats for Player 2
    print("\nCollecting stats for Player 2:")
    player2_name = input("Enter Player 2 Name: ").strip()
    player2_stats_list = collect_player_stats(2)

    # Compute average stats for each player
    try:
        player1_avg_stats = compute_average_stats(player1_stats_list, player1_name)
        player2_avg_stats = compute_average_stats(player2_stats_list, player2_name)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Get number of simulations
    try:
        num_simulations_input = input("\nEnter the number of simulations to run (default 1000): ").strip()
        num_simulations = int(num_simulations_input) if num_simulations_input else 1000
        if num_simulations < 1:
            raise ValueError
    except ValueError:
        print("Invalid number of simulations. Using default of 1000.")
        num_simulations = 1000

    # Predict the winner
    winner, win_rate, confidence_interval = predict_match(
        player1_avg_stats, player2_avg_stats, num_simulations
    )

    # Display results
    print(f"\nMonte Carlo simulation results ({num_simulations} iterations):")
    print(f"{player1_avg_stats['name']} Win Rate: {win_rate * 100:.2f}%")
    print(f"{player2_avg_stats['name']} Win Rate: {100 - win_rate * 100:.2f}%")
    print(
        f"95% Confidence Interval for {player1_avg_stats['name']}: {confidence_interval[0]*100:.2f}% - {confidence_interval[1]*100:.2f}%"
    )
    print(f"Predicted Winner: {winner}")