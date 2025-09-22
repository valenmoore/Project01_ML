"""
Trains the CoverageNetwork class to recognize offensive formation (I-Form, Shotgun, etc.) based on defensive positioning
"""

import os
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from data_compiler import DataCompiler
from coverage_network import CoverageNetwork
import constants

dc = DataCompiler()

# get all play tracking for defensive players before snap
position_filter = constants.D_POSITIONS
movements = dc.get_all_plays_movement(position_filter=position_filter, before_snap=True)
wanted_keys = ['x', 'y', 's', 'a', 'o', 'dir']  # values to pass into network
num_features = len(wanted_keys) + len(constants.D_POSITIONS)
X, y = [], []  # X stores player positions and other features, y stores labels

for play in movements:
    # add to dataset for each play
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

    play_features = []
    play_labels = []
    for frame_df in play:
        # get the features to add to X
        # features are x, y, speed, acceleration, orientation rotation, movement direction, and player position in one hot format
        features = frame_df[wanted_keys].to_numpy()  # features from wanted_keys

        position_features = []
        for _, player in frame_df.iterrows():
            # add one hot position
            pos_idx = constants.D_POSITIONS.index(player['position'])
            one_hot = np.zeros(len(constants.D_POSITIONS))
            one_hot[pos_idx] = 1
            position_features.append(one_hot)
        position_features = np.array(position_features)

        features = np.hstack([features, position_features])  # stack features and one hot

        play_features.append(features)

    X.append(play_features)  # an array of plays, where each play is an array of frames (will be flattened afterwards)

    # use the index of the offensive formation as the label
    label = constants.OFFENSE_FORMATIONS.index(play_data["offenseFormation"].iloc[0])
    y.append(label)

print("Completed constructing dataset.")

# only use 50 frames before snap to prevent the model being confused by huddle or other pre-snap formations
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

# flatten out the dataset so it is just an array of frames not associated with plays
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

# split the x and y into train and test (80% train)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_classes = len(constants.OFFENSE_FORMATIONS)
model = CoverageNetwork(num_classes=len(constants.OFFENSE_FORMATIONS))

# compile the model with an Adam optimizer and SparseCC for softmax predictions
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# saves the model every epoch
save_path = "./models/offense_formation"
model_index = len(os.listdir(save_path))
checkpoint_cb = ModelCheckpoint(
    f"{save_path}/{model_index}/model.keras",  # file to save model
    save_best_only=True,  # only save when validation loss improves
    monitor='val_loss',  # metric to monitor
    mode='min',  # minimize val_loss
    verbose=1
)

# fit the model to to the train and test data, saving every epoch
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=50,
    callbacks=[checkpoint_cb]
)
