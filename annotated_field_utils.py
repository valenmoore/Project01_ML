import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrow
from matplotlib import animation
from scipy.ndimage import zoom
from annotated_play import AnnotatedPlay

class AnnotatedField:
    """
    A class to display a frame-by-frame annotated field, including safety zones, blitzers, and marking matchups
    Pairs with the AnnotatedPlay class.
    """

    def __init__(self, field_length=120, field_width=53.3, interval=100):
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
        self.hm = None

        self.annotated_play: AnnotatedPlay | None = None

    def set_and_rescale_hm(self, heatmap):
        # Field dimensions
        field_length = 120.0
        field_width = 53.3

        # Heatmap dimensions
        heatmap_rows, heatmap_cols = heatmap.shape

        # Rescale heatmap to field dimensions
        scale_x = field_length / heatmap_cols
        scale_y = field_width / heatmap_rows

        # Rescaling using interpolation
        rescaled_heatmap = zoom(heatmap, (scale_y, scale_x), order=1)
        self.hm = rescaled_heatmap

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
        """

        :param game_id:
        :param play_id:
        :return:
        """
        self.player_play_data = self.dc.get_all_player_play_data(game_id, play_id)

    def plot_play_animation(self, play: AnnotatedPlay):
        # self.plot_player_heatmap(play, "SS", color="Reds")
        # self.plot_player_heatmap(play, "CB", color="Blues")
        # self.plot_heatmap_on_field()
        self.annotated_play = play
        ani = animation.FuncAnimation(self.fig, self.update_plot_play, frames=len(play), interval=self._interval,
                                      repeat=True, fargs=(play,))
        plt.show()

    def update_side_text(self, frame_info):
        # Clear previous side text without resetting axis properties
        self.axs[1].cla()  # Use `cla()` instead of `clear()`

        # Set axis limits and ensure we work in the proper range for transAxes (0 to 1)
        self.axs[1].set_xlim(0, 1)
        self.axs[1].set_ylim(0, 1)

        y_pos = 0.98  # Starting y-position (scaled to 0 to 1 range)
        line_spacing = -0.03  # Space between lines (scaled to 0 to 1 range)

        off_formation = frame_info["off_formation"]
        correct = off_formation == self.annotated_play.real_off_formation
        self.axs[1].text(0.5, y_pos,
                         f"Prediction: {off_formation}\n(actual: {self.annotated_play.real_off_formation})",
                         ha='center', va='top', fontweight='bold', fontsize=10, color="green" if correct else "red",
                         transform=self.axs[1].transAxes, zorder=5)

        coverage = frame_info["coverage"]
        correct = coverage == self.annotated_play.real_coverage
        self.axs[1].text(0.5, y_pos + line_spacing * 2,
                         f"Prediction: {coverage}\n(actual: {self.annotated_play.real_coverage})",
                         ha='center', va='top', fontweight='bold', fontsize=10, color="green" if correct else "red",
                         transform=self.axs[1].transAxes, zorder=5)

        self.axs[1].text(0.5, y_pos + line_spacing * 4, "Predicted Safeties:",
                         ha='center', va='top', fontweight='bold', fontsize=10, color="black",
                         transform=self.axs[1].transAxes, zorder=5)

        player_rows = frame_info["players"]
        y_pos = y_pos + line_spacing * 5
        for nfl_id in frame_info["deep_safeties"]:
            player = player_rows[player_rows["nflId"] == nfl_id].iloc[0]
            name = player['displayName']
            conf = float(frame_info["deep_safeties_conf"][nfl_id])
            conf = round(conf, 4) * 100
            self.axs[1].text(0.5, y_pos, f"{name} #{player['jerseyNumber']} ({conf:.2f}%):",
                             ha='center', va='top', fontweight='bold', fontsize=10, color="black",
                             transform=self.axs[1].transAxes, zorder=5)

            self.axs[1].text(0.5, y_pos + line_spacing, f"s: {player['s']} y/s, a: {player['a']} y/s²",
                             ha='center', va='top', fontweight='bold', fontsize=9, color="black",
                             transform=self.axs[1].transAxes, zorder=5)

            self.axs[1].text(0.5, y_pos + 2 * line_spacing, f"Position: {player['position']}",
                             ha='center', va='top', fontweight='bold', fontsize=9, color="black",
                             transform=self.axs[1].transAxes, zorder=5)

            y_pos += 4 * line_spacing



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
        frame_info = play[frame]
        player_rows = frame_info["players"]
        self.plot_players(player_rows)

        # Update ball position if it's part of the frame data
        self.update_ball(player_rows)
        # self.plot_heatmap_on_field()
        self.update_side_text(frame_info)
        self.plot_deep_safeties(frame_info, play.final_deep_safeties)
        self.plot_blitzers(frame_info)
        self.plot_man_defenders(frame_info)
        self._update_field_title(f"Game {player_rows.iloc[0]['gameId']} Play {player_rows.iloc[0]['playId']} -- Frame {frame}")

    def plot_deep_safeties(self, frame_info, safety_ids):
        players = frame_info["players"]

        safety_ids = frame_info["deep_safeties"]
        if not safety_ids or len(safety_ids) == 0:
            return

        num_safeties = len(safety_ids)

        for safety_id in safety_ids:
            safety_row = players[players["nflId"] == safety_id]
            if safety_row.empty:
                continue

            x, y = safety_row.iloc[0]["x"], safety_row.iloc[0]["y"]

            # Oval dimensions scale with number of safeties
            fw, fl = self.field_width, self.field_length
            if num_safeties == 1:  # Cover 1: very wide single oval
                width, height = fl * 0.05, fw * 0.6
            elif num_safeties == 2:  # Cover 2: medium ovals
                width, height = fl * 0.05, fw * 0.4
            elif num_safeties == 3:  # Cover 3: narrower thirds
                width, height = fl * 0.05, fw * 0.25
            elif num_safeties == 4:  # Quarters
                width, height = fl * 0.05, fw * 0.15
            else:  # Default
                width, height = 30, 15

            # Create ellipse centered on safety’s position
            oval = patches.Ellipse(
                (x, y),
                width=width,
                height=height,
                edgecolor="#424026",
                facecolor="#f7ea34",
                lw=2,
                ls="--",
                alpha=0.5,
                zorder=1
            )
            self.axs[0].add_patch(oval)
            self.misc.append(oval)  # store for clearing next frame

    def plot_blitzers(self, frame_info):
        players = frame_info["players"]
        blitzers = frame_info["blitzers"]
        for blitz_id in blitzers:
            player_row = players[players["nflId"] == blitz_id]
            if player_row.empty:
                continue

            x, y = player_row.iloc[0]["x"], player_row.iloc[0]["y"]
            arrow_length = -5  # length of arrow in yards
            if player_row["playDirection"].iloc[0] == "left":
                arrow_length = -arrow_length
            arrow = FancyArrow(
                x, y,  # starting point
                arrow_length, 0,  # dx, dy
                width=0.1,  # thickness of the arrow
                color="red",  # arrow color
                zorder=3
            )
            self.axs[0].add_patch(arrow)
            self.misc.append(arrow)

    def plot_man_defenders(self, frame_info):
        players = frame_info["players"]
        man_defs = frame_info["man"]
        for pid in man_defs:
            player_row = players[players["nflId"] == pid]
            if player_row.empty:
                continue

            x, y = player_row.iloc[0]["x"], player_row.iloc[0]["y"]
            arrow_length = -2  # length of arrow in yards
            if player_row["playDirection"].iloc[0] == "left":
                arrow_length = -arrow_length
            arrow = FancyArrow(
                x, y,  # starting point
                arrow_length, 0,  # dx, dy
                width=0.1,  # thickness of the arrow
                color="black",  # arrow color
                zorder=3
            )
            self.axs[0].add_patch(arrow)
            self.misc.append(arrow)


    def plot_players(self, player_rows, show_numbers=True, show_names=True):
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

    def plot_heatmap_on_field(self):
        self.axs[0].imshow(self.hm, extent=(0, 120, 0, 53, 3),
                           origin='lower', alpha=0.6, cmap='coolwarm', zorder=2)
