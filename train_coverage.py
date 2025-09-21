import os

from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from data_compiler import DataCompiler
from coverage_network import CoverageNetwork
import numpy as np
import constants
from sklearn.model_selection import train_test_split
import tensorflow as tf

dc = DataCompiler()

position_filter = constants.D_POSITIONS
movements = dc.get_all_plays_movement(position_filter=position_filter, before_snap=True)
wanted_keys = ['x', 'y', 's', 'a', 'o', 'dir']  # values to pass into network
num_features = len(wanted_keys) + len(constants.D_POSITIONS)
X, y = [], []
# ["MAN", "2R", "2L", "3R", "3M", "3L", "4OR", "4OL", "4IR", "4IL", "FR", "FL", "HCR", "HCL", "CFR", "CFL", "HOL", "DF", "PRE"]
coverage_assignments = ["Man", "Cover 2", "Cover 3", "Quarters", "Underneath", "Deep"]
coverage_map = {
    "Man": ["MAN"],
    "Cover 2": ["2R", "2L"],
    "Cover 3": ["3R", "3M", "3L"],
    "Quarters": ["4OR", "4OL", "4IR", "4IL"],
    "Underneath": ["FR", "FL", "HCR", "HCL", "CFR", "CFL", "HOL"],
    "Deep": ["DF", "PRE"]
}
for play in movements:
    first_frame = play[0]
    game_id = first_frame['gameId'].iloc[0]
    play_id = first_frame['playId'].iloc[0]

    play_data = dc.get_play_data_by_id(game_id, play_id)
    if not play_data["pff_passCoverage"].iloc[0] in constants.COVERAGE_FORMATIONS:
        continue  # skip na plays
    if not play_data["pff_passCoverage"].iloc[0] in constants.COMMON_COVERAGE_FORMATIONS:
        # skip non cover 1, 2, 3, or quarters
        continue

    player_play_data = dc.get_all_player_play_data(game_id, play_id)

    """    man_zone_str = play_data['pff_manZone'].iloc[0]
    if man_zone_str not in constants.MAN_ZONE:
        continue
    man_zone_label = constants.MAN_ZONE.index(man_zone_str)"""
    play_features = []
    play_labels = []
    for frame_df in play:
        features = frame_df[wanted_keys].to_numpy()

        position_features = []
        for _, player in frame_df.iterrows():
            pos_idx = constants.D_POSITIONS.index(player['position'])
            one_hot = np.zeros(len(constants.D_POSITIONS))
            one_hot[pos_idx] = 1
            position_features.append(one_hot)
        position_features = np.array(position_features)

        features = np.hstack([features, position_features])

        play_features.append(features)

    """frame_labels = []
    # get the labels from one of the rows by iterating through players
    for _, row in play[0].iterrows():
        player_id = row['nflId']
        player_row = player_play_data[player_play_data['nflId'] == player_id]
        if player_row.empty:
            label = 0
        else:
            # coverage assignment as integer
            cov_str = player_row['pff_defensiveCoverageAssignment'].iloc[0]
            label = 0  # default to 0
            if cov_str in constants.PLAYER_COVERAGE_ASSIGNMENTS:
                # label = constants.PLAYER_COVERAGE_ASSIGNMENTS.index(cov_str) + 1
                for k in coverage_map:
                    if cov_str in coverage_map[k]:
                        label = coverage_assignments.index(k) + 1
                label = 2  # for zone
                if cov_str == "MAN":
                    label = 1

        frame_labels.append(label)
        # print(len(frame_labels))
    if len(frame_labels) != len(position_filter):
        print("frame_labels length not equal to position_filter at", game_id, play_id)
        continue"""

    X.append(play_features)
    # y.append(frame_labels)

    label = constants.COMMON_COVERAGE_FORMATIONS.index(play_data["pff_passCoverage"].iloc[0])
    y.append(label)


print("Completed constructing dataset.")
max_len = 50
print(sum([len(play) for play in X]) / len(X))

num_players = len(position_filter)
X_padded = np.zeros((len(X), max_len, num_players, num_features))
for i, play in enumerate(X):
    if len(play) > max_len:
        play = play[-max_len:]

    for t, frame in enumerate(play):
        # frame might have < 22 players; pad missing rows with zeros
        n_players = frame.shape[0]
        X_padded[i, t, :n_players, :] = frame[:num_players, :]

X = np.array(X_padded)

flat_Y = []
flat_X = []
for i, play in enumerate(X):
    label = y[i]
    for frame in play:
        flat_X.append(frame)
        flat_Y.append(label)

X = np.array(flat_X)
num_players = len(position_filter)

print(X.shape)
y = flat_Y
y = np.array(y, dtype=np.int32)
print(y.shape)
# y_one_hot = tf.one_hot(y, depth=len(coverage_assignments) + 1)

X_train_gnn, X_test_gnn, y_train_gnn, y_test_gnn = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_classes = len(constants.COMMON_COVERAGE_FORMATIONS)
model = CoverageNetwork(num_classes=len(constants.COMMON_COVERAGE_FORMATIONS))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
save_path = "./models/player_coverage"
model_index = len(os.listdir(save_path))
checkpoint_cb = ModelCheckpoint(
    f"{save_path}/{model_index}/model.keras",  # file to save model
    save_best_only=True,  # only save when validation loss improves
    monitor='val_loss',  # metric to monitor
    mode='min',  # minimize val_loss
    verbose=1
)

model.fit(
    X_train_gnn,
    y_train_gnn,
    validation_data=(X_test_gnn, y_test_gnn),
    batch_size=32,
    epochs=50,
    callbacks=[checkpoint_cb]
)
