import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from classifier_network import Classifier
from data_compiler import DataCompiler


class Field:
    def __init__(self, classifier: Classifier, dc: DataCompiler, field_length=120, field_width=53.3, interval=100):
        self.classifier = classifier
        self.dc = dc
        self.field_length = field_length
        self.field_width = field_width
        self.plot, self.fig, self.axs = self.create_field()
        self.home_color = "#f74036"
        self.away_color = "#3d19b3"
        self.player_play_data = None
        self._interval = interval  # Interval in milliseconds for the animation timer
        self.player_scatters = []  # To hold player scatter objects
        self.jersey_texts = []  # To hold jersey number text objects
        self.name_texts = []
        self.misc = []
        self.ball_marker = None  # To hold the ball marker for animation
        self.side_text = None
        self.special_player_id = None
        self.coverage_pairs = {}

    def show(self):
        self.plot.show()

    def _update_field_title(self, new_title):
        self.axs[0].set_title(new_title)

    def create_field(self):
        # Create field setup
        fig, axs = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [3, 1]})
        axs[0].set_xlim(0, self.field_length)
        axs[0].set_ylim(0, self.field_width)
        field_color = "#22a82d"

        # Field outline
        axs[0].add_patch(patches.Rectangle((0, 0), self.field_length, self.field_width, linewidth=2,
                                           edgecolor=field_color, facecolor=field_color))

        # Yard lines and hashes (simplified)
        for x in range(20, 110, 10):
            axs[0].plot([x, x], [0, self.field_width], color="white", linewidth=1)
            axs[0].text(x, 1, str(x - 10 if x <= 50 else 110 - x), color="white", fontsize=15, ha='center',
                        fontweight="bold")
            axs[0].text(x, self.field_width - 2, str(x - 10 if x <= 50 else 110 - x), color="white", fontsize=15,
                        ha='center', fontweight="bold")

        end_zone_color = "#30a9cf"

        # End zones
        axs[0].add_patch(
            patches.Rectangle((0, 0), 10, self.field_width, linewidth=2, facecolor=end_zone_color))
        axs[0].add_patch(
            patches.Rectangle((110, 0), 10, self.field_width, linewidth=2, facecolor=end_zone_color))

        axs[0].set_facecolor(field_color)
        axs[0].set_title("Field Display")  # generic
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        # Set up the side axis for displaying additional information
        axs[1].axis("off")  # Turn off the axis
        axs[1].set_title("Play Information")

        plt.tight_layout()

        return plt, fig, axs

    def load_player_play_data(self, game_id, play_id):
        self.player_play_data = self.dc.get_all_player_play_data(game_id, play_id)

    def plot_play_animation(self, play):
        ani = animation.FuncAnimation(self.fig, self.update_plot_play, frames=len(play), interval=self._interval,
                                      repeat=True, fargs=(play,))
        plt.show()

    def update_side_text(self, player_rows):
        # Clear previous side text without resetting axis properties
        self.axs[1].cla()  # Use `cla()` instead of `clear()`

        # Set axis limits and ensure we work in the proper range for transAxes (0 to 1)
        self.axs[1].set_xlim(0, 1)
        self.axs[1].set_ylim(0, 1)

        y_pos = 0.98  # Starting y-position (scaled to 0 to 1 range)
        line_spacing = -0.03  # Space between lines (scaled to 0 to 1 range)
        player_rows = [p for _, p in player_rows.iterrows()]
        pred = self.classifier.predict_man_zone(player_rows)
        real = self.dc.get_play_data_by_id(player_rows[0]['gameId'], player_rows[0]['playId'])
        self.axs[1].text(0.5, y_pos, f"Prediction: {pred} (actual: {real['pff_manZone'].iloc[0]})",
                         ha='center', va='top', fontweight='bold', fontsize=10, color="black",
                         transform=self.axs[1].transAxes, zorder=5)

        form = self.classifier.predict_formation(player_rows)
        self.axs[1].text(0.5, y_pos + line_spacing, f"Prediction: {form} (actual: {real['offenseFormation'].iloc[0]})",
                         ha='center', va='top', fontweight='bold', fontsize=10, color="black",
                         transform=self.axs[1].transAxes, zorder=5)

        cov, conf = self.classifier.predict_coverage(player_rows)
        self.axs[1].text(0.5, y_pos + line_spacing * 2,
                         f"Prediction: {cov} - {round(conf)}% conf. \n(actual: {real['pff_passCoverage'].iloc[0]})",
                         ha='center', va='top', fontweight='bold', fontsize=10, color="black",
                         transform=self.axs[1].transAxes, zorder=5)

        # Loop through player data and add text with rectangles
        """for player in player_rows:
            print(player.keys())
            if player['routeRan']:  # Only display if there's a route
                # Display Player's Jersey Number
                name = player['displayName'].split(" ")[1].upper()
                self.axs[1].text(0.5, y_pos, f"{name} #{player['jerseyNumber']} ({player['position']}):",
                                 ha='center', va='top', fontweight='bold', fontsize=10, color="black",
                                 transform=self.axs[1].transAxes, zorder=5)

                # Display Speed and Acceleration
                self.axs[1].text(0.5, y_pos + line_spacing, f"s: {player['s']} y/s, a: {player['a']} y/s²",
                                 ha='center', va='top', fontweight='bold', fontsize=9, color="black",
                                 transform=self.axs[1].transAxes, zorder=5)

                # Display Route description
                self.axs[1].text(0.5, y_pos + 2 * line_spacing, f"Route: {player['route']}",
                                 ha='center', va='top', fontweight='bold', fontsize=9, color="black",
                                 transform=self.axs[1].transAxes, zorder=5)


                # Update y_pos for the next player's info
                y_pos += 5 * line_spacing  # Move down by the height of the text + spacing"""

    def update_plot_play(self, frame, play):
        # Clear previous players
        for scatter in self.player_scatters:
            scatter.remove()
        self.player_scatters.clear()

        for text in self.jersey_texts:
            text.remove()
        self.jersey_texts.clear()

        for text in self.name_texts:
            text.remove()
        self.name_texts.clear()

        for m in self.misc:
            m.remove()
        self.misc.clear()

        # Plot players for the current frame
        player_rows = play[frame]
        self.plot_players(player_rows)

        # Update ball position if it's part of the frame data
        self.update_ball(player_rows)
        self.update_side_text(player_rows)
        self._update_field_title(f"Game {player_rows.iloc[0]['gameId']} Play {player_rows.iloc[0]['playId']} -- Frame {frame}")

    def plot_players(self, player_rows, show_numbers=True, show_names=True, plot_coverage=False):
        teams = []
        for i, row in player_rows.iterrows():
            if row['club'] not in teams:
                teams.append(row['club'])
                if len(teams) > 2:
                    break
        current_positions = {}
        for i, row in player_rows.iterrows():
            pos = (row['x'], row['y'])
            if row['displayName'] != "football":
                current_positions[row['nflId']] = pos
                color = self.home_color if row['club'] == teams[0] else self.away_color
                # print(row['club'], color)
                """if row['nflId'] == self.special_player_id:
                    color = "#e8eb34"
                """

                scatter = self.axs[0].scatter(*pos, color=color, s=100, edgecolor='black', zorder=2)
                self.player_scatters.append(scatter)
                if show_names:
                    name = row['displayName']
                    # SETS NAME TO COVERAGE
                    nfl_id = int(row['nflId'])
                    if self.player_play_data is not None:
                        name = self.player_play_data[self.player_play_data['nflId'] == nfl_id]['pff_defensiveCoverageAssignment'].iloc[0]
                    name_text = self.axs[0].text(pos[0], pos[1] - 1, name, color=color, fontsize=5,
                                                 ha="center",
                                                 va="center", fontweight="bold", zorder=2)
                    self.name_texts.append(name_text)
                if show_numbers:
                    text = self.axs[0].text(pos[0], pos[1], row['jerseyNumber'], color="white", fontsize=5, ha="center",
                                            va="center", zorder=2)
                    self.jersey_texts.append(text)

        if plot_coverage:
            for d_id in self.coverage_pairs:
                pos_1 = current_positions[d_id]
                pos_2 = current_positions[self.coverage_pairs[d_id]]
                line = self.axs[0].plot([pos_1[0], pos_2[0]], [pos_1[1], pos_2[1]], color='purple', linestyle='-')
                self.misc.extend(line)

    def update_ball(self, player_rows):
        # Find the ball in player rows
        ball_data = next((row for _, row in player_rows.iterrows() if row['displayName'] == "Football"), None)
        if ball_data:
            ball_pos = (ball_data['x'], ball_data['y'])
            # Update or create the ball marker
            if self.ball_marker is None:
                self.ball_marker = patches.Ellipse(ball_pos, width=1, height=0.5, edgecolor='brown',
                                                   facecolor='saddlebrown', lw=2, zorder=3)
                self.axs[0].add_patch(self.ball_marker)
            else:
                # Update position of existing ball marker
                self.ball_marker.set_center(ball_pos)
