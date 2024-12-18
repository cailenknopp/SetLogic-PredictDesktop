from app import Player, Match
import statistics

def predict_winner(player1_avg_stats, player2_avg_stats, num_simulations=1000):
    """
    Simulate multiple matches to predict the winner between two players using averaged stats.
    """
    player1_wins = 0
    player2_wins = 0

    # Run simulations
    for _ in range(num_simulations):
        # Create Player instances
        player1 = Player(**player1_avg_stats)
        player2 = Player(**player2_avg_stats)

        # Create a Match instance
        match = Match(player1, player2)

        # Simulate the match
        winner_name = match.simulate()

        # Tally wins
        if winner_name == player1.name:
            player1_wins += 1
        else:
            player2_wins += 1

    # Calculate win rates
    player1_win_rate = player1_wins / num_simulations
    player2_win_rate = player2_wins / num_simulations

    # Predict the winner
    predicted_winner = player1_avg_stats['name'] if player1_win_rate > player2_win_rate else player2_avg_stats['name']

    # Display results
    print(f"\n{player1_avg_stats['name']} Win Rate: {player1_win_rate * 100:.2f}%")
    print(f"{player2_avg_stats['name']} Win Rate: {player2_win_rate * 100:.2f}%")
    print(f"Predicted Winner: {predicted_winner}")

if __name__ == "__main__":
    # Function to collect stats for a player
    def collect_player_stats(player_number):
        stats_list = []
        while True:
            print(f"\nEnter statistics for Player {player_number} (Game #{len(stats_list) + 1}):")
            stats = {}
            stats['serve_percentage'] = float(input("Serve Percentage (0 to 1): "))
            stats['break_point_conversion'] = float(input("Break Point Conversion (0 to 1): "))
            stats['first_serve_won'] = float(input("First Serve Won (0 to 1): "))
            stats['second_serve_won'] = float(input("Second Serve Won (0 to 1): "))
            stats_list.append(stats)
            cont = input("Add another game's stats for this player? (y/n): ")
            if cont.lower() != 'y':
                break
        return stats_list

    # Collect stats for Player 1
    print("Collecting stats for Player 1:")
    player1_name = input("Enter Player 1 Name: ")
    player1_stats_list = collect_player_stats(1)

    # Collect stats for Player 2
    print("\nCollecting stats for Player 2:")
    player2_name = input("Enter Player 2 Name: ")
    player2_stats_list = collect_player_stats(2)

    # Function to compute average stats
    def compute_average_stats(stats_list):
        averaged_stats = {}
        for key in ['serve_percentage', 'break_point_conversion', 'first_serve_won', 'second_serve_won']:
            averaged_stats[key] = statistics.mean([stats[key] for stats in stats_list])
        averaged_stats['sets_won'] = 0
        averaged_stats['games_won'] = 0
        averaged_stats['points_won'] = 0
        return averaged_stats

    # Compute average stats for each player
    player1_avg_stats = compute_average_stats(player1_stats_list)
    player1_avg_stats['name'] = player1_name

    player2_avg_stats = compute_average_stats(player2_stats_list)
    player2_avg_stats['name'] = player2_name

    # Get number of simulations
    num_simulations = int(input("\nEnter the number of simulations to run: "))

    # Predict the winner
    predict_winner(player1_avg_stats, player2_avg_stats, num_simulations)