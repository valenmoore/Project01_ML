"""
Trains the PlayerCoverageNetwork class to predict player-by-player labels, like safeties, blitzers, and man defenders
"""
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split

from player_coverage_network import PlayerCoverageNetwork
from data_compiler import DataCompiler
import constants

model = PlayerCoverageNetwork()
dc = DataCompiler()

position_filter = constants.D_POSITIONS
movements = dc.get_all_plays_movement(position_filter=position_filter, before_snap=True)
wanted_keys = ['x', 'y', 's', 'a', 'o', 'dir']  # values to pass into network
num_features = len(wanted_keys) + len(constants.D_POSITIONS)
X, y = [], []

# deep safety coverage names
deep_coverages = ["DF", "PRE", "2R", "2L", "3R", "3M", "3L", "4OR", "4OL", "4IR", "4IL"]

# the coverage to detect for (will only look at plays from that coverage)
# a separate model is trained for each of the four common coverages
current_coverage = "Cover-1"
# choose either BLITZ, MAN, or SAFETIES for training task
current_task = "BLITZ"

for play in movements:
    first_frame = play[0]
    game_id = first_frame['gameId'].iloc[0]
    play_id = first_frame['playId'].iloc[0]

    play_data = dc.get_play_data_by_id(game_id, play_id)
    if play_data["pff_passCoverage"].iloc[0] != current_coverage:
        continue  # skip plays not matching current coverage

    player_play_data = dc.get_all_player_play_data(game_id, play_id)

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

    frame_labels = []
    # get the labels from one of the rows by iterating through players
    for _, row in play[0].iterrows():
        player_id = row['nflId']
        player_row = player_play_data[player_play_data['nflId'] == player_id]
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
                if current_task == "SAFETIES":
                    if cov_str in deep_coverages:
                        # player is deep safety
                        label = 1
                if current_task == "MAN":
                    # set label to 1 if player is a man defender
                    if cov_str == "MAN":
                        label = 1
                elif current_task == "BLITZ":
                    label = 0  # if the player has a coverage assignment, they are not a blitzer

        frame_labels.append(label)

    if len(frame_labels) != len(position_filter):
        print("frame_labels length not equal to position_filter at", game_id, play_id)
        continue

    if current_task == "SAFETIES":
        # skip any plays with no safeties
        if not 1 in frame_labels:
            continue

    X.append(play_features)
    y.append(frame_labels)

max_len = 50

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


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# binary crossentropy because each player has a binary label
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

save_path = f"./models/{current_coverage}_player/"
model_index = len(os.listdir(save_path))
checkpoint_cb = ModelCheckpoint(
    f"{save_path}/{model_index}/model.keras",  # file to save model
    save_best_only=True,  # only save when validation loss improves
    monitor='val_loss',  # metric to monitor
    mode='min',  # minimize val_loss
    verbose=1
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=100,
    callbacks=[checkpoint_cb]
)