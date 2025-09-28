"""
Trains the PlayerCoverageNetwork class to predict player-by-player labels, like safeties, blitzers, and man defenders
"""
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.metrics import Precision, Recall

from offense_aware_player_coverage import PlayerCoverageNetwork
from data_compiler import DataCompiler
import constants

model = PlayerCoverageNetwork()
dc = DataCompiler()

position_filter = constants.POSITIONS
movements = dc.get_all_plays_movement(position_filter=position_filter, before_snap=True)
wanted_keys = ['x', 'y', 's', 'a', 'o', 'dir']  # values to pass into network
X_d, X_o, y = [], [], []

# deep safety coverage names
deep_coverages = ["DF", "PRE", "2R", "2L", "3R", "3M", "3L", "4OR", "4OL", "4IR", "4IL"]

# the coverage to detect for (will only look at plays from that coverage)
# a separate model is trained for each of the four common coverages
current_coverage = "Cover-1"
# choose either BLITZ, MAN, for training task
current_task = "MAN"

for play in movements:
    first_frame = play[0]
    game_id = first_frame['gameId'].iloc[0]
    play_id = first_frame['playId'].iloc[0]

    play_data = dc.get_play_data_by_id(game_id, play_id)
    if play_data["pff_passCoverage"].iloc[0] != current_coverage:
        continue  # skip plays not matching current coverage

    player_play_data = dc.get_all_player_play_data(game_id, play_id)

    play_features_d = []
    play_features_o = []
    play_labels = []
    for frame_df in play:
        features = frame_df[wanted_keys].to_numpy()

        position_features_d = []
        position_features_o = []
        o_features = []
        d_features = []
        for _, player in frame_df.iterrows():
            raw_features = player[wanted_keys].to_numpy()
            if player['position'] in constants.D_POSITIONS:
                pos_idx = constants.D_POSITIONS.index(player['position'])
                one_hot = np.zeros(len(constants.D_POSITIONS))
                one_hot[pos_idx] = 1
                d_features.append(np.hstack([raw_features, one_hot]))
            else:
                pos_idx = constants.O_POSITIONS.index(player['position'])
                one_hot = np.zeros(len(constants.O_POSITIONS))
                one_hot[pos_idx] = 1
                o_features.append(np.hstack([raw_features, one_hot]))

        play_features_d.append(np.array(d_features))
        play_features_o.append(np.array(o_features))

    frame_labels = []
    # get the labels from one of the rows by iterating through players
    for _, row in play[0].iterrows():
        player_id = row['nflId']
        player_row = player_play_data[player_play_data['nflId'] == player_id]
        if row["position"] not in constants.D_POSITIONS:
            continue
        if player_row.empty:
            label = 0
        else:
            # coverage assignment as integer
            cov_str = player_row['pff_defensiveCoverageAssignment'].iloc[0]
            if current_task == "BLITZ":
                label = 1  # default to 1 for blitzers because only players with no coverage assignment are blitzing
            else:
                label = 0  # default to 0
            if cov_str in constants.PLAYER_COVERAGE_ASSIGNMENTS:
                if current_task == "MAN":
                    # set label to 1 if player is a man defender
                    if cov_str == "MAN":
                        label = 1
                elif current_task == "BLITZ":
                    label = 0  # if the player has a coverage assignment, they are not a blitzer

        frame_labels.append(label)

    if len(frame_labels) != len(constants.D_POSITIONS):
        print("frame_labels length not equal to position_filter at", game_id, play_id)
        continue

    X_d.append(play_features_d)
    X_o.append(play_features_o)
    y.append(frame_labels)

max_len = 50

num_players = 11
num_features_d = len(wanted_keys) + len(constants.D_POSITIONS)
num_features_o = len(wanted_keys) + len(constants.O_POSITIONS)
X_padded_d = np.zeros((len(X_d), max_len, num_players, num_features_d))
X_padded_o = np.zeros((len(X_o), max_len, num_players, num_features_o))
for i, (play_d, play_o) in enumerate(zip(X_d, X_o)):
    if len(play_d) > max_len:
        play_d = play_d[-max_len:]
        play_o = play_o[-max_len:]

    for t, (frame_d, frame_o) in enumerate(zip(play_d, play_o)):
        n_players_d = frame_d.shape[0]
        n_players_o = frame_o.shape[0]

        X_padded_d[i, t, :n_players_d, :] = frame_d[:11, :]
        X_padded_o[i, t, :n_players_o, :] = frame_o[:11, :]

X_d = np.array(X_padded_d)
X_o = np.array(X_padded_o)

flat_Y = []
flat_X_d = []
flat_X_o = []
for i, play in enumerate(X_d):
    label = y[i]
    for frame in play:
        flat_X_d.append(frame)
        flat_Y.append(label)

for play in X_o:
    for frame in play:
        flat_X_o.append(frame)

X_d = np.array(flat_X_d)
X_o = np.array(flat_X_o)
num_players = len(position_filter)

print(X_d.shape)
y = flat_Y
y = np.array(y, dtype=np.int32)
print(y.shape)
# y_one_hot = tf.one_hot(y, depth=len(coverage_assignments) + 1)

all_labels = np.concatenate(y)

classes = np.unique(all_labels)

# compute class weights to balance biases
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=all_labels,
)

class_weight_dict = dict(zip(classes, class_weights))

print("Unique classes:", classes)
print("Computed class weights (array):", class_weights)
print("Computed class weights (dictionary):", class_weight_dict)

X_d_train, X_d_test, X_o_train, X_o_test, y_train, y_test = train_test_split(
    X_d, X_o, y, test_size=0.2, random_state=42
)

# binary crossentropy because each player has a binary label
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

if current_task == "BLITZ":
    save_path = f"./models/blitz_models"
else:
    save_path = f"./models/man_models"

checkpoint_cb = ModelCheckpoint(
    f"{save_path}/{current_coverage}.keras",  # file to save model
    save_best_only=True,  # only save when validation loss improves
    monitor='val_loss',  # metric to monitor
    mode='min',  # minimize val_loss
    verbose=1
)

model.fit(
    (X_d_train, X_o_train),
    y_train,
    validation_data=((X_d_test, X_o_test), y_test),
    batch_size=32,
    epochs=100,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_cb]
)