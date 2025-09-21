import numpy as np
from field_utils import Field
from data_compiler import DataCompiler
from classifier_network import Classifier
from play_analyzer import PlayAnalyzer
from annotated_play import AnnotatedPlay
from annotated_field_utils import AnnotatedField
import random
import constants

dc = DataCompiler()
classifier = Classifier(dc)
field = Field(classifier, dc)
# play = dc.get_play_by_id(75)
# dc.merge_add_position_to_data()
"""classifier.train_coverage_classifier()
f = dc.get_team_movement(position_filter=constants.D_POSITIONS, before_snap=True)
correct = 0
incorrect = 0
for play in f[:50]:
    for frame in play:
        try:
            cls = classifier.predict_coverage(frame)
            real = dc.get_play_data_by_id(frame[0]['gameId'], frame[0]['playId'])['pff_passCoverage'].iloc[0]
            if real == cls:
                correct += 1
            else:
                incorrect += 1
        except Exception as e:
            print(e)
print(correct, "correct", incorrect, "incorrect")
print(correct / (correct + incorrect), "correct")"""

game = random.choice(dc.game_ids)
# f = dc.get_team_movement(position_filter=constants.O_POSITIONS, before_snap=True)
play_ids = dc.get_play_ids_from_game_id(game)
pid = random.choice(play_ids)
info = dc.get_play_data_by_id(game, pid)
# while info['pff_manZone'].iloc[0] != "Man" or (info['passResult'].iloc[0] != "C" and info['passResult'].iloc[0] != "I"):
while info['pff_passCoverage'].iloc[0] not in ["Cover-1", "Cover-2", "Cover-3", "Quarters"] or (info['passResult'].iloc[0] != "C" and info['passResult'].iloc[0] != "I"):
    game = random.choice(dc.game_ids)
    play_ids = dc.get_play_ids_from_game_id(game)
    pid = random.choice(play_ids)
    info = dc.get_play_data_by_id(game, pid)
    print(info['passResult'].iloc[0])

play = dc.get_play_tracking_by_id(game, pid)
play_info = dc.get_play_data_by_id(game, pid)
analyzer = PlayAnalyzer()
annotated_field = AnnotatedField()
an_play = AnnotatedPlay(dc, analyzer, game, pid)
annotated_field.plot_play_animation(an_play)
"""new_play = []
d_player_poses = {}
o_player_poses = {}
full_d_positions = {}
qb_positions = []
for frame in play:
    for i, player in frame.iterrows():
        if player['position'] in ["WR", "TE", "RB"]:
            o_player_poses.setdefault(player['nflId'], []).append((player['x'], player['y']))
        elif player['position'] in ['CB', 'FS', 'SS', 'OLB', 'MLB', 'LB']:
            d_player_poses.setdefault(player['nflId'], []).append((player['x'], player['y']))
        if player['position'] in constants.D_POSITIONS:
            full_d_positions.setdefault(player['nflId'], []).append((player['x'], player['y']))
        elif player['position'] == "QB":
            qb_positions.append((player['x'], player['y']))"""



"""ids = hm_net.detect_blitz(full_d_positions, qb_positions, play_info['absoluteYardlineNumber'].iloc[0])
for frame in play:
    for player in frame:
        if player['nflId'] in ids:
            player['club'] = play_info['possessionTeam'].iloc[0]"""

"""o_ids = list(o_player_poses.keys())
o_poses = list(o_player_poses.values())
for d_id in d_player_poses:
    player_poses = d_player_poses[d_id]
    coverage, probabilities = hm_net.get_player_coverage_probabilities(player_poses, o_poses)
    if coverage != "ZONE":
        cover_index = np.argmax(probabilities)
        cover_id = o_ids[cover_index]
        field.coverage_pairs[d_id] = cover_id

field.load_player_play_data(game, pid)
field.plot_play_animation(play)"""
"""nfl_id = 0
pos = None
for f in play:
    for p in f:
        if p['position'] == 'SS':
            nfl_id = p['nflId']
            pos = p['position']
bs = []
for f in play:
    for p in f:
        if p['nflId'] == nfl_id and p['frameType'] == "BEFORE_SNAP":
            bs.append((p['x'], p['y']))
hm = hm_net.generate_heatmap(bs, hm_net.grid_shape)
field.axs[1].imshow(hm, cmap='hot', interpolation='nearest')
field.plot_play_animation(play)"""
"""nfl_id = 0
pos = None
for f in play:
    for p in f:
        if p['position'] == 'LB' or p['position'] == "MLB" or p['position'] == "OLB":
            nfl_id = p['nflId']
            pos = p['position']
bs = []
for f in play:
    for p in f:
        if p['nflId'] == nfl_id and p['frameType'] == "BEFORE_SNAP":
            bs.append(p)
pred = hm_net.predict(bs)
after_s = []
for f in play:
    for p in f:
        if p['nflId'] == nfl_id and p['frameType'] == "AFTER_SNAP":
            after_s.append(p)
poses = [(p['x'], p['y']) for p in after_s]
field.special_player_id = nfl_id
hm = hm_net.generate_heatmap(poses, hm_net.grid_shape)
print(hm_net.model.summary())

def plot_heatmaps(predicted_heatmap, real_heatmap):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Predicted Heatmap
    axes[0].imshow(predicted_heatmap, cmap='hot', interpolation='nearest')
    axes[0].set_title('Predicted Heatmap')
    axes[0].axis('off')  # Optional: Remove axes for better visualization

    # Real Heatmap
    axes[1].imshow(real_heatmap, cmap='hot', interpolation='nearest')
    axes[1].set_title('Real Heatmap')
    axes[1].axis('off')  # Optional: Remove axes for better visualization

    plt.tight_layout()
    plt.show()


# plot_heatmaps(pred, hm)
rows = []
for f in play:
    for p in f:
        if p['nflId'] == nfl_id:
            rows.append([p])
print(rows)
field.set_and_rescale_hm(pred)
field.plot_play_animation(rows)"""

# hm = hm_net.generate_heatmap(bs, (60, 27))
# print(hm_net.predict(bs))
# field.plot_play_animation(play)
# classifier.train_coverage_classifier()
# field.plot_players(rows)
# field.show()
