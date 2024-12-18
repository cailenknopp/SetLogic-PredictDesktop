# app.py

from flask import Flask, render_template, request, flash
import random

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure key in production

class Player:
    def __init__(self, name, serve_percentage, break_point_conversion, first_serve_won, second_serve_won,
                 sets_won=0, games_won=0, points_won=0):
        self.name = name
        self.serve_percentage = serve_percentage  # Probability of serve being in (0 to 1)
        self.break_point_conversion = break_point_conversion
        self.first_serve_won = first_serve_won  # Probability of winning the point on first serve (0 to 1)
        self.second_serve_won = second_serve_won  # Probability of winning the point on second serve (0 to 1)
        self.sets_won = sets_won
        self.games_won = games_won
        self.points_won = points_won

class Match:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    def simulate_point(self, server, receiver):
        serve_in = random.random() < server.serve_percentage
        if serve_in:
            # First serve
            point_won = random.random() < server.first_serve_won
        else:
            # Second serve
            point_won = random.random() < server.second_serve_won
        return server if point_won else receiver

    def simulate_game(self, server, receiver):
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

    def simulate_tie_break(self, server, receiver):
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

    def simulate_set(self, server, receiver):
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
                return self.player1
            if games_p2 >= 6 and games_p2 - games_p1 >= 2:
                return self.player2

            # Handle tie-break at 6-6
            if games_p1 == 6 and games_p2 == 6:
                tie_break_winner = self.simulate_tie_break(current_server, current_receiver)
                return tie_break_winner

            # Alternate server for next game
            current_server, current_receiver = current_receiver, current_server

    def simulate_match_state(self):
        simulations = 1000
        p1_wins = 0
        p2_wins = 0

        for _ in range(simulations):
            # Clone players for simulation
            sim_player1 = Player(
                name=self.player1.name,
                serve_percentage=self.player1.serve_percentage,
                break_point_conversion=self.player1.break_point_conversion,
                first_serve_won=self.player1.first_serve_won,
                second_serve_won=self.player1.second_serve_won
            )
            sim_player2 = Player(
                name=self.player2.name,
                serve_percentage=self.player2.serve_percentage,
                break_point_conversion=self.player2.break_point_conversion,
                first_serve_won=self.player2.first_serve_won,
                second_serve_won=self.player2.second_serve_won
            )
            sim_match = Match(sim_player1, sim_player2)

            # Initialize sets won
            sets_p1 = self.player1.sets_won
            sets_p2 = self.player2.sets_won

            # Randomly decide who serves first
            if random.random() < 0.5:
                server = sim_player1
                receiver = sim_player2
            else:
                server = sim_player2
                receiver = sim_player1

            while sets_p1 < 2 and sets_p2 < 2:
                set_winner = sim_match.simulate_set(server, receiver)
                if set_winner == sim_player1:
                    sets_p1 += 1
                else:
                    sets_p2 += 1

                # Alternate server for next set
                server, receiver = receiver, server

            if sets_p1 >= 2:
                p1_wins += 1
            else:
                p2_wins += 1

        probability_p1 = p1_wins / simulations
        probability_p2 = p2_wins / simulations

        return {
            'player1': {
                'name': self.player1.name,
                'prob': probability_p1,
                'simulations_won': p1_wins
            },
            'player2': {
                'name': self.player2.name,
                'prob': probability_p2,
                'simulations_won': p2_wins
            }
        }

    def simulate(self):
        return self.simulate_match_state()

@app.route('/', methods=['GET', 'POST'])
def index():
    probabilities = None
    if request.method == 'POST':
        form = request.form
        try:
            player1_stats = {
                'name': form['player1_name'],
                'serve_percentage': float(form['player1_serve']) / 100,
                'break_point_conversion': float(form['player1_break']) / 100,
                'first_serve_won': float(form['player1_first_serve_won']) / 100,
                'second_serve_won': float(form['player1_second_serve_won']) / 100,
                'sets_won': int(form['player1_sets_won']),
                'games_won': int(form['player1_games_won']),
                'points_won': int(form['player1_points_won'])
            }
            player2_stats = {
                'name': form['player2_name'],
                'serve_percentage': float(form['player2_serve']) / 100,
                'break_point_conversion': float(form['player2_break']) / 100,
                'first_serve_won': float(form['player2_first_serve_won']) / 100,
                'second_serve_won': float(form['player2_second_serve_won']) / 100,
                'sets_won': int(form['player2_sets_won']),
                'games_won': int(form['player2_games_won']),
                'points_won': int(form['player2_points_won'])
            }

            player1 = Player(**player1_stats)
            player2 = Player(**player2_stats)
            match = Match(player1, player2)
            probabilities = match.simulate()

        except KeyError as e:
            missing_field = e.args[0]
            flash(f"Missing form field: {missing_field}")
        except ValueError as e:
            flash(f"Invalid input: {e}")

    return render_template('index.html', probabilities=probabilities)

if __name__ == '__main__':
    app.run(debug=True)