import numpy as np
from data_compiler import DataCompiler
from coverage_network import CoverageNetwork
from coverage_network import CoverageNetwork
from player_coverage_network import PlayerCoverageNetwork
from field_utils import Field
from classifier_network import Classifier
import random
import constants
from tensorflow.keras.models import load_model
import tensorflow as tf

def masked_sparse_cce(y_true, y_pred):
    mask = tf.not_equal(y_true, 0)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)

def non_zero_accuracy(y_true, y_pred):
    """Accuracy only on non-zero classes (ignoring class 0)"""
    # Always cast y_true to int32
    y_true_int = tf.cast(y_true, tf.int32)

    # Get predictions as int32
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    # Create mask for non-zero labels
    mask = tf.not_equal(y_true_int, 0)

    # Get correct predictions
    correct_predictions = tf.equal(predictions, y_true_int)

    # Apply mask and compute accuracy
    masked_correct = tf.boolean_mask(correct_predictions, mask)

    # Use tf.cond to handle empty case
    return tf.cond(
        tf.size(masked_correct) > 0,
        lambda: tf.reduce_mean(tf.cast(masked_correct, tf.float32)),
        lambda: 0.0
    )


dc = DataCompiler()
classifier = Classifier(dc)
field = Field(classifier, dc)

current_coverage = "Cover-1"
ids = dc.get_game_ids_by_coverage(current_coverage)
selected_pair = ids.sample(n=1)
game_id, play_id = selected_pair["gameId"].iloc[0], selected_pair["playId"].iloc[0]
print(game_id, play_id)
# f = dc.get_team_movement(position_filter=constants.O_POSITIONS, before_snap=True)
info = dc.get_play_data_by_id(game_id, play_id)
play = dc.get_play_tracking_by_id(game_id, play_id)
play_info = dc.get_play_data_by_id(game_id, play_id)
model = load_model("./models/player_coverage/19/model.keras", custom_objects={"GNNCoverageNetwork": CoverageNetwork})
cov_model = load_model(f"models/{current_coverage}_player/2/model.keras", custom_objects={"PlayerCoverageNetwork": PlayerCoverageNetwork})

play_data = dc.get_play_data_by_id(game_id, play_id)
player_play_data = dc.get_all_player_play_data(game_id, play_id)

print("Selected play")
print(play_data["pff_passCoverage"])

wanted_keys = ['x', 'y', 's', 'a', 'o', 'dir']  # values to pass into network

players_arr = []
print(play)
for _, player in play[0].iterrows():
    if player["position"] in constants.D_POSITIONS:
        players_arr.append(player)

for frame in play:
    print(frame.iloc[0]["frameType"])
    if frame.iloc[0]["frameType"] != "BEFORE_SNAP":
        break
    model_inputs = []
    for _, player in frame.iterrows():
        p_data = player_play_data[player_play_data['nflId'] == player["nflId"]]
        if player["position"] in constants.D_POSITIONS:
            pos_idx = constants.D_POSITIONS.index(player['position'])
            one_hot = np.zeros(len(constants.D_POSITIONS))
            one_hot[pos_idx] = 1

            features = player[wanted_keys].to_numpy()

            features = np.hstack([features, one_hot])
            model_inputs.append(features)

    inputs = np.array(model_inputs, dtype=np.float32)
    inputs = np.expand_dims(inputs, 0)
    # inputs = np.expand_dims(inputs, axis=0)
    preds = model(inputs)
    pred_labels = np.argmax(preds)
    print("Coverage", constants.COVERAGE_FORMATIONS[pred_labels])
    pred_player = cov_model(inputs).numpy().squeeze()
    top_indices = np.argsort(pred_player)[-4:]  # top 2 players
    for i in top_indices:
        player = players_arr[i]
        print(player[["displayName", "position"]])
        print(
            player_play_data[player_play_data['nflId'] == player["nflId"]]["pff_defensiveCoverageAssignment"].iloc[0])


field.plot_play_animation(play)
