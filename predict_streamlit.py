import streamlit as st
import numpy as np
import random
from typing import List, Dict, Tuple
from scipy import stats  # For confidence intervals
import matplotlib.pyplot as plt
import io

# Inject CSS for light mode theme
st.markdown(
    """
    <style>
    .main {
        background-color: #FFFFFF;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

def main():
    st.title("slPredict - Tennis Monte Carlo Simulation")
    st.write("Run Monte Carlo simulations to predict the winner of a tennis match, using the recent statistics of both players.")   
    num_simulations = st.number_input(
        "Number of Simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )

    # Player 1
    st.header("Player 1 Details")
    p1_name = st.text_input("Player 1 Name", "Player 1")
    num_matches_p1 = st.number_input(
        "Number of Matches for Player 1",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key='num_matches_p1'
    )

    p1_stats_list = []
    st.subheader(f"Enter Stats for {p1_name}")
    for i in range(int(num_matches_p1)):
        st.markdown(f"**Match {i+1}**")
        serve_percentage = st.number_input(
            f"Serve Percentage (Match {i+1})",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01,
            format="%.2f",
            key=f"p1_serve_{i}"
        )
        break_point_conversion = st.number_input(
            f"Break Point Conversion (Match {i+1})",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            format="%.2f",
            key=f"p1_break_{i}"
        )
        first_serve_won = st.number_input(
            f"First Serve Won (%) (Match {i+1})",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
            format="%.2f",
            key=f"p1_first_{i}"
        )
        second_serve_won = st.number_input(
            f"Second Serve Won (%) (Match {i+1})",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            format="%.2f",
            key=f"p1_second_{i}"
        )

        match_stats = {
            'serve_percentage': serve_percentage,
            'break_point_conversion': break_point_conversion,
            'first_serve_won': first_serve_won,
            'second_serve_won': second_serve_won
        }
        p1_stats_list.append(match_stats)

    # Player 2
    st.header("Player 2 Details")
    p2_name = st.text_input("Player 2 Name", "Player 2")
    num_matches_p2 = st.number_input(
        "Number of Matches for Player 2",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key='num_matches_p2'
    )

    p2_stats_list = []
    st.subheader(f"Enter Stats for {p2_name}")
    for i in range(int(num_matches_p2)):
        st.markdown(f"**Match {i+1}**")
        serve_percentage = st.number_input(
            f"Serve Percentage (Match {i+1})",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01,
            format="%.2f",
            key=f"p2_serve_{i}"
        )
        break_point_conversion = st.number_input(
            f"Break Point Conversion (Match {i+1})",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            format="%.2f",
            key=f"p2_break_{i}"
        )
        first_serve_won = st.number_input(
            f"First Serve Won (%) (Match {i+1})",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
            format="%.2f",
            key=f"p2_first_{i}"
        )
        second_serve_won = st.number_input(
            f"Second Serve Won (%) (Match {i+1})",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            format="%.2f",
            key=f"p2_second_{i}"
        )

        match_stats = {
            'serve_percentage': serve_percentage,
            'break_point_conversion': break_point_conversion,
            'first_serve_won': first_serve_won,
            'second_serve_won': second_serve_won
        }
        p2_stats_list.append(match_stats)

    if st.button("Predict Match"):
        try:
            player1_avg_stats = compute_average_stats(p1_stats_list, p1_name)
            player2_avg_stats = compute_average_stats(p2_stats_list, p2_name)

            predicted_winner, win_rate, confidence_interval = predict_match(
                player1_avg_stats, player2_avg_stats, int(num_simulations)
            )

            st.subheader("Prediction Results")
            st.write(f"Predicted Winner: **{predicted_winner}**")
            st.write(f"{player1_avg_stats['name']} Win Rate: {win_rate * 100:.2f}%")
            st.write(f"{player2_avg_stats['name']} Win Rate: {100 - win_rate * 100:.2f}%")
            st.write(
                f"95% Confidence Interval for {player1_avg_stats['name']} Win Rate: "
                f"{confidence_interval[0]*100:.2f}% - {confidence_interval[1]*100:.2f}%"
            )

            # Generate and display the bar chart
            fig, ax = plt.subplots()
            players = [player1_avg_stats['name'], player2_avg_stats['name']]
            win_rates = [win_rate * 100, (100 - win_rate * 100)]
            colors = ['blue', 'orange']

            ax.bar(players, win_rates, color=colors)
            ax.set_ylabel('Win Rate (%)')
            ax.set_title('Match Prediction Win Rates')
            for i, v in enumerate(win_rates):
                ax.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')
            st.pyplot(fig)

            # Save the plot to a bytes buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            # Prepare the report content
            report = f"""
**Tennis Match Prediction Report**

**Number of Simulations:** {num_simulations}

**Player 1: {player1_avg_stats['name']}**
- Serve Percentage: {player1_avg_stats['serve_percentage']:.2f}
- Break Point Conversion: {player1_avg_stats['break_point_conversion']:.2f}
- First Serve Won: {player1_avg_stats['first_serve_won']:.2f}
- Second Serve Won: {player1_avg_stats['second_serve_won']:.2f}

**Player 2: {player2_avg_stats['name']}**
- Serve Percentage: {player2_avg_stats['serve_percentage']:.2f}
- Break Point Conversion: {player2_avg_stats['break_point_conversion']:.2f}
- First Serve Won: {player2_avg_stats['first_serve_won']:.2f}
- Second Serve Won: {player2_avg_stats['second_serve_won']:.2f}

**Prediction Results**
- Predicted Winner: {predicted_winner}
- {player1_avg_stats['name']} Win Rate: {win_rate * 100:.2f}%
- {player2_avg_stats['name']} Win Rate: {100 - win_rate * 100:.2f}%
- 95% Confidence Interval for {player1_avg_stats['name']} Win Rate: {confidence_interval[0]*100:.2f}% - {confidence_interval[1]*100:.2f}%
"""

            # Add the download button for the report
            st.download_button(
                label="Download Results Report",
                data=report,
                file_name="tennis_match_prediction_report.txt",
                mime="text/plain"
            )

            # Add the download button for the graph
            st.download_button(
                label="Download Win Rates Graph",
                data=buf,
                file_name="win_rates_graph.png",
                mime="image/png"
            )

        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()