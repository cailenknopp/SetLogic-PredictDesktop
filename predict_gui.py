import numpy as np
import random
import statistics
from typing import List, Dict, Tuple
from scipy import stats  # For confidence intervals
import customtkinter as ctk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Initialize CustomTkinter
ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"


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
) -> Tuple[str, float, Tuple[float, float], Dict]:
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

    # Collect histogram data
    win_rates = {
        'Player 1': player1_wins,
        'Player 2': player2_wins
    }

    return predicted_winner, player1_win_rate, confidence_interval, win_rates


class PredictGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Tennis Match Predictor")
        self.geometry("800x800")
        self.resizable(False, False)

        # Create Tabs
        self.tabview = ctk.CTkTabview(self, width=780, height=780)
        self.tabview.pack(pady=10, padx=10)
        self.tabview.add("Players")
        self.tabview.add("Simulation")

        self.tab_players = self.tabview.tab("Players")
        self.tab_simulation = self.tabview.tab("Simulation")

        # Players Tab
        self.create_players_tab()

        # Simulation Tab
        self.create_simulation_tab()

    def create_players_tab(self):
        # Create a Scrollable Frame within the Players tab
        self.scrollable_frame = ctk.CTkScrollableFrame(self.tab_players, corner_radius=10, width=760, height=780)
        self.scrollable_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Container Frame to center content
        container_frame = ctk.CTkFrame(self.scrollable_frame, corner_radius=10)
        container_frame.pack(pady=20, padx=20, fill="x", expand=True)

        # Player 1 Frame
        self.player1_frame = ctk.CTkFrame(container_frame, corner_radius=10)
        self.player1_frame.pack(pady=10, padx=40, fill="both", expand=True)

        # Player 1 Inputs
        self.player1_name_var = ctk.StringVar()
        self.player1_label = ctk.CTkLabel(self.player1_frame, text="Player 1", font=("Arial", 16))
        self.player1_label.pack(pady=10)

        self.player1_name = ctk.CTkEntry(self.player1_frame, textvariable=self.player1_name_var, placeholder_text="Name", width=200)
        self.player1_name.pack(pady=5, padx=20, anchor="center")  # Centered entry

        # Set trace to update label dynamically
        self.player1_name_var.trace_add("write", lambda *args: self.update_player_label(1))

        self.player1_stats_frames = []
        self.player1_stats_list = []

        # Add Game Stats for Player 1
        add_game1_button = ctk.CTkButton(self.player1_frame, text="Add Game", command=lambda: self.add_game_stats(1))
        add_game1_button.pack(pady=5, anchor="center")  # Centered button

        # Separator
        separator1 = ctk.CTkLabel(container_frame, text="")
        separator1.pack(pady=20)

        # Player 2 Frame
        self.player2_frame = ctk.CTkFrame(container_frame, corner_radius=10)
        self.player2_frame.pack(pady=10, padx=40, fill="both", expand=True)

        # Player 2 Inputs
        self.player2_name_var = ctk.StringVar()
        self.player2_label = ctk.CTkLabel(self.player2_frame, text="Player 2", font=("Arial", 16))
        self.player2_label.pack(pady=10)

        self.player2_name = ctk.CTkEntry(self.player2_frame, textvariable=self.player2_name_var, placeholder_text="Name", width=200)
        self.player2_name.pack(pady=5, padx=20, anchor="center")  # Centered entry

        # Set trace to update label dynamically
        self.player2_name_var.trace_add("write", lambda *args: self.update_player_label(2))

        self.player2_stats_frames = []
        self.player2_stats_list = []

        # Add Game Stats for Player 2
        add_game2_button = ctk.CTkButton(self.player2_frame, text="Add Game", command=lambda: self.add_game_stats(2))
        add_game2_button.pack(pady=5, anchor="center")  # Centered button

    def update_player_label(self, player_number: int):
        if player_number == 1:
            name = self.player1_name_var.get().strip()
            if name:
                self.player1_label.configure(text=name)
            else:
                self.player1_label.configure(text="Player 1")
        elif player_number == 2:
            name = self.player2_name_var.get().strip()
            if name:
                self.player2_label.configure(text=name)
            else:
                self.player2_label.configure(text="Player 2")

    def add_game_stats(self, player_number: int):
        if player_number == 1:
            parent_frame = self.player1_frame
            stats_frames = self.player1_stats_frames
            stats_list = self.player1_stats_list
            player_label = "Player 1"
        else:
            parent_frame = self.player2_frame
            stats_frames = self.player2_stats_frames
            stats_list = self.player2_stats_list
            player_label = "Player 2"

        game_frame = ctk.CTkFrame(parent_frame, corner_radius=8, fg_color="transparent")
        game_frame.pack(pady=5, padx=40, fill="x")  # Increased padx for centering

        game_label = ctk.CTkLabel(game_frame, text=f"Game {len(stats_frames)+1}", font=("Arial", 12, "bold"))
        game_label.grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky="w")

        serve_percentage = ctk.CTkEntry(game_frame, placeholder_text="Serve Percentage (0-100)", width=150)
        serve_percentage.grid(row=1, column=0, padx=10, pady=2, sticky="e")

        break_conversion = ctk.CTkEntry(game_frame, placeholder_text="Break Point Conversion (0-100)", width=150)
        break_conversion.grid(row=1, column=1, padx=10, pady=2, sticky="w")

        first_serve = ctk.CTkEntry(game_frame, placeholder_text="First Serve Won (0-100)", width=150)
        first_serve.grid(row=2, column=0, padx=10, pady=2, sticky="e")

        second_serve = ctk.CTkEntry(game_frame, placeholder_text="Second Serve Won (0-100)", width=150)
        second_serve.grid(row=2, column=1, padx=10, pady=2, sticky="w")

        remove_button = ctk.CTkButton(game_frame, text="X", fg_color="red", width=25, height=25, command=lambda: self.remove_game_stats(stats_frames, game_frame))
        remove_button.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        stats_frames.append(game_frame)

    def remove_game_stats(self, stats_frames, game_frame):
        game_frame.destroy()
        stats_frames.remove(game_frame)

    def create_simulation_tab(self):
        # Simulation Controls
        controls_frame = ctk.CTkFrame(self.tab_simulation, corner_radius=10)
        controls_frame.pack(pady=20, padx=20, fill="x")

        simulations_label = ctk.CTkLabel(controls_frame, text="Number of Simulations:", font=("Arial", 14))
        simulations_label.pack(pady=10)  # Centered label by default

        self.num_simulations = ctk.CTkEntry(controls_frame, placeholder_text="e.g., 1000", width=200)
        self.num_simulations.pack(pady=5, anchor="center")  # Smaller and centered entry

        run_button = ctk.CTkButton(controls_frame, text="Run Simulation", command=self.run_simulation)
        run_button.pack(pady=10)  # Reduced padding for better layout

        export_button = ctk.CTkButton(controls_frame, text="Export Summary", command=self.export_summary)
        export_button.pack(pady=10)  # Export button

        # Results Display
        results_frame = ctk.CTkFrame(self.tab_simulation, corner_radius=10)
        results_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # Average Stats Labels
        self.results_labels_frame = ctk.CTkFrame(results_frame, corner_radius=10)
        self.results_labels_frame.pack(pady=10, padx=10, fill="x")

        self.player1_win_rate_label = ctk.CTkLabel(self.results_labels_frame, text="", font=("Arial", 12))
        self.player1_win_rate_label.pack(pady=5)

        self.player2_win_rate_label = ctk.CTkLabel(self.results_labels_frame, text="", font=("Arial", 12))
        self.player2_win_rate_label.pack(pady=5)

        self.confidence_interval_label = ctk.CTkLabel(self.results_labels_frame, text="", font=("Arial", 12))
        self.confidence_interval_label.pack(pady=5)

        self.predicted_winner_label = ctk.CTkLabel(self.results_labels_frame, text="", font=("Arial", 14, "bold"))
        self.predicted_winner_label.pack(pady=10)

        # Average Stats for Player 1
        self.player1_avg_stats_label = ctk.CTkLabel(self.results_labels_frame, text="", font=("Arial", 12, "bold"))
        self.player1_avg_stats_label.pack(pady=5)

        # Average Stats for Player 2
        self.player2_avg_stats_label = ctk.CTkLabel(self.results_labels_frame, text="", font=("Arial", 12, "bold"))
        self.player2_avg_stats_label.pack(pady=5)

        # Graph Display
        self.graph_frame = ctk.CTkFrame(results_frame, corner_radius=10)
        self.graph_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.ax.set_title("Simulation Results")
        self.ax.set_xlabel("Players")
        self.ax.set_ylabel("Number of Wins")

    def gather_player_stats(self, player_number: int) -> List[Dict]:
        if player_number == 1:
            stats_frames = self.player1_stats_frames
            player_name = self.player1_name.get().strip()
        else:
            stats_frames = self.player2_stats_frames
            player_name = self.player2_name.get().strip()

        stats_list = []
        for frame in stats_frames:
            try:
                serve_percentage_entry = frame.grid_slaves(row=1, column=0)[0]
                break_conversion_entry = frame.grid_slaves(row=1, column=1)[0]
                first_serve_entry = frame.grid_slaves(row=2, column=0)[0]
                second_serve_entry = frame.grid_slaves(row=2, column=1)[0]

                serve_percentage = float(serve_percentage_entry.get()) / 100
                break_conversion = float(break_conversion_entry.get()) / 100
                first_serve = float(first_serve_entry.get()) / 100
                second_serve = float(second_serve_entry.get()) / 100

                if not (0 <= serve_percentage <= 1 and 0 <= break_conversion <= 1 and
                        0 <= first_serve <= 1 and 0 <= second_serve <= 1):
                    raise ValueError

                game_stats = {
                    'serve_percentage': serve_percentage,
                    'break_point_conversion': break_conversion,
                    'first_serve_won': first_serve,
                    'second_serve_won': second_serve
                }
                stats_list.append(game_stats)
            except:
                messagebox.showerror("Input Error", f"Please enter valid numerical values between 0 and 100 for all game stats of {'Player 1' if player_number == 1 else 'Player 2'}.")
                return []

        if not player_name:
            messagebox.showerror("Input Error", f"Please enter the name for {'Player 1' if player_number == 1 else 'Player 2'}.")
            return []

        if not stats_list:
            messagebox.showerror("Input Error", f"Please add at least one game stat for {'Player 1' if player_number == 1 else 'Player 2'}.")
            return []

        return stats_list

    def run_simulation(self):
        # Gather Player 1 Stats
        player1_stats = self.gather_player_stats(1)
        if not player1_stats:
            return
        player1_name = self.player1_name.get().strip()

        # Gather Player 2 Stats
        player2_stats = self.gather_player_stats(2)
        if not player2_stats:
            return
        player2_name = self.player2_name.get().strip()

        # Compute Average Stats
        try:
            player1_avg_stats = compute_average_stats(player1_stats, player1_name)
            player2_avg_stats = compute_average_stats(player2_stats, player2_name)
        except ValueError as e:
            messagebox.showerror("Computation Error", str(e))
            return

        # Get Number of Simulations
        try:
            num_simulations_input = self.num_simulations.get()
            num_simulations = int(num_simulations_input) if num_simulations_input else 1000
            if num_simulations < 1:
                raise ValueError
        except:
            messagebox.showerror("Input Error", "Please enter a valid positive integer for the number of simulations.")
            return

        # Run Prediction
        try:
            winner, win_rate, confidence_interval, win_rates = predict_match(
                player1_avg_stats, player2_avg_stats, num_simulations
            )
        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))
            return

        # Display Results
        self.player1_win_rate_label.configure(text=f"{player1_avg_stats['name']} Win Rate: {win_rate * 100:.2f}%")
        self.player2_win_rate_label.configure(text=f"{player2_avg_stats['name']} Win Rate: {100 - win_rate * 100:.2f}%")
        self.confidence_interval_label.configure(text=f"95% Confidence Interval for {player1_avg_stats['name']}: {confidence_interval[0]*100:.2f}% - {confidence_interval[1]*100:.2f}%")
        self.predicted_winner_label.configure(text=f"Predicted Winner: {winner}")

        # Display Average Stats
        player1_stats_text = (
            f"{player1_avg_stats['name']} Average Stats:\n"
            f" Serve Percentage: {player1_avg_stats['serve_percentage']*100:.2f}%\n"
            f" Break Point Conversion: {player1_avg_stats['break_point_conversion']*100:.2f}%\n"
            f" First Serve Won: {player1_avg_stats['first_serve_won']*100:.2f}%\n"
            f" Second Serve Won: {player1_avg_stats['second_serve_won']*100:.2f}%"
        )
        self.player1_avg_stats_label.configure(text=player1_stats_text)

        player2_stats_text = (
            f"{player2_avg_stats['name']} Average Stats:\n"
            f" Serve Percentage: {player2_avg_stats['serve_percentage']*100:.2f}%\n"
            f" Break Point Conversion: {player2_avg_stats['break_point_conversion']*100:.2f}%\n"
            f" First Serve Won: {player2_avg_stats['first_serve_won']*100:.2f}%\n"
            f" Second Serve Won: {player2_avg_stats['second_serve_won']*100:.2f}%"
        )
        self.player2_avg_stats_label.configure(text=player2_stats_text)

        # Update Graph
        self.ax.clear()
        self.ax.set_title("Simulation Results")
        self.ax.set_xlabel("Players")
        self.ax.set_ylabel("Number of Wins")
        players = [player1_avg_stats['name'], player2_avg_stats['name']]
        wins = [win_rates['Player 1'], win_rates['Player 2']]
        self.ax.bar(players, wins, color=['blue', 'green'])
        self.canvas.draw()

    def export_summary(self):
        """Export a 600x600 summary PNG with the graph and answers displayed."""
        # Retrieve current simulation results
        player1_text = self.player1_win_rate_label.cget("text")
        player2_text = self.player2_win_rate_label.cget("text")
        confidence_text = self.confidence_interval_label.cget("text")
        winner_text = self.predicted_winner_label.cget("text")
        player1_avg_text = self.player1_avg_stats_label.cget("text")
        player2_avg_text = self.player2_avg_stats_label.cget("text")

        # Create a matplotlib figure
        fig = Figure(figsize=(6, 6), dpi=100)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # Plot the simulation graph
        players = [player1_avg_stats['name'] for player1_avg_stats in [
            compute_average_stats(self.gather_player_stats(1), self.player1_name.get().strip())
        ]][0:1] + [player2_avg_stats['name'] for player2_avg_stats in [
            compute_average_stats(self.gather_player_stats(2), self.player2_name.get().strip())
        ]][0:1]
        wins = [self.player1_win_rate_label.cget("text").split(': ')[1].replace('%', ''),
                self.player2_win_rate_label.cget("text").split(': ')[1].replace('%', '')]

        try:
            wins = [int(float(win)) * 10 for win in wins]  # Scaled for visibility
        except:
            wins = [0, 0]  # Default if parsing fails

        ax1.bar(players, wins, color=['blue', 'green'])
        ax1.set_title("Simulation Results")
        ax1.set_xlabel("Players")
        ax1.set_ylabel("Number of Wins")

        # Add simulation details as text
        details = f"{player1_text}\n{player2_text}\n{confidence_text}\n{winner_text}\n\n{player1_avg_text}\n\n{player2_avg_text}"
        ax2.text(0.5, 0.5, details, horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=10)
        ax2.axis('off')  # Hide the axes

        # Adjust layout
        fig.tight_layout()

        # Save the figure as a PNG
        try:
            fig.savefig("simulation_summary.png")
            messagebox.showinfo("Export Successful", "Summary exported as 'simulation_summary.png'.")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting: {e}")


if __name__ == "__main__":
    app = PredictGUI()
    app.mainloop()