import streamlit as st
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
        server_points = 0
        receiver_points = 0
        while True:
            point_winner = self.simulate_point(server, receiver)
            if point_winner == server:
                server_points += 1
            else:
                receiver_points += 1

            # Simple game win condition
            if server_points >= 4 and server_points - receiver_points >= 2:
                server.games_won += 1
                return server
            if receiver_points >= 4 and receiver_points - server_points >= 2:
                receiver.games_won += 1
                return receiver

    def simulate_set(self, server: Player, receiver: Player) -> Player:
        while True:
            game_winner = self.simulate_game(server, receiver)
            if game_winner == server:
                server.sets_won += 1
            else:
                receiver.sets_won += 1

            # Simple set win condition
            if server.sets_won >= 6 and server.sets_won - receiver.sets_won >= 2:
                return server
            if receiver.sets_won >= 6 and receiver.sets_won - server.sets_won >= 2:
                return receiver

    def run_simulation(self) -> Player:
        current_server = self.player1
        current_receiver = self.player2
        while True:
            set_winner = self.simulate_set(current_server, current_receiver)
            if set_winner == self.player1:
                return self.player1
            else:
                return self.player2


def main():
    st.title("Tennis Match Simulator")

    st.header("Player 1 Details")
    p1_name = st.text_input("Name", "Player 1")
    p1_serve = st.slider("Serve Percentage", 0.0, 1.0, 0.6)
    p1_break = st.slider("Break Point Conversion", 0.0, 1.0, 0.3)
    p1_first = st.slider("First Serve Won (%)", 0.0, 1.0, 0.7)
    p1_second = st.slider("Second Serve Won (%)", 0.0, 1.0, 0.5)

    st.header("Player 2 Details")
    p2_name = st.text_input("Name", "Player 2", key="p2")
    p2_serve = st.slider("Serve Percentage", 0.0, 1.0, 0.55, key="serve_p2")
    p2_break = st.slider("Break Point Conversion", 0.0, 1.0, 0.25, key="break_p2")
    p2_first = st.slider("First Serve Won (%)", 0.0, 1.0, 0.65, key="first_p2")
    p2_second = st.slider("Second Serve Won (%)", 0.0, 1.0, 0.45, key="second_p2")

    if st.button("Simulate Match"):
        player1 = Player(p1_name, p1_serve, p1_break, p1_first, p1_second)
        player2 = Player(p2_name, p2_serve, p2_break, p2_first, p2_second)
        match = Match(player1, player2)
        winner = match.run_simulation()
        st.write(f"The winner is {winner.name}!")


if __name__ == "__main__":
    main()