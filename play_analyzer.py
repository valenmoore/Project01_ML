from tensorflow.keras.models import load_model
import numpy as np

import constants
from coverage_network import CoverageNetwork
from player_coverage_network import PlayerCoverageNetwork

class PlayAnalyzer:
    def __init__(self):
        self.coverage_model_path = "./models/player_coverage/19/model.keras"
        self.coverage_model = self._load_coverage_model()
        self.player_models = self._load_player_models()
        self.blitz_model = load_model("./models/Cover-1_player/3/model.keras", custom_objects={"PlayerCoverageNetwork": PlayerCoverageNetwork})
        self.man_model = load_model("./models/Cover-1_player/4/model.keras", custom_objects={"PlayerCoverageNetwork": PlayerCoverageNetwork})

    def _load_coverage_model(self):
        return load_model(self.coverage_model_path, custom_objects={"GNNCoverageNetwork": CoverageNetwork})

    def _load_player_models(self):
        coverages = ["Cover-1", "Cover-2", "Cover-3", "Quarters"]
        models = {}
        for coverage in coverages:
            path = f"./models/{coverage}_player/best/model.keras"
            models[coverage] = load_model(path, custom_objects={"PlayerCoverageNetwork": PlayerCoverageNetwork})
        return models

    def _construct_model_inputs(self, frame):
        wanted_keys = ['x', 'y', 's', 'a', 'o', 'dir']  # values to pass into network

        model_inputs = []
        for _, player in frame.iterrows():
            if player["position"] in constants.D_POSITIONS:
                pos_idx = constants.D_POSITIONS.index(player['position'])
                one_hot = np.zeros(len(constants.D_POSITIONS))
                one_hot[pos_idx] = 1

                features = player[wanted_keys].to_numpy()

                features = np.hstack([features, one_hot])
                model_inputs.append(features)

        inputs = np.array(model_inputs, dtype=np.float32)
        inputs = np.expand_dims(inputs, 0)
        return inputs

    def analyze_frame(self, frame, return_confs=False):
        inputs = self._construct_model_inputs(frame)
        coverage_preds = self.coverage_model(inputs)
        coverage_pred = np.argmax(coverage_preds)
        coverage = constants.COMMON_COVERAGE_FORMATIONS[coverage_pred]

        num_players_map = {
            "Cover-1": 1,
            "Cover-2": 2,
            "Cover-3": 3,
            "Quarters": 4
        }

        num_players = num_players_map[coverage]
        pred_safeties = self.player_models[coverage](inputs).numpy().squeeze()
        top_indices = np.argsort(pred_safeties)[-num_players:]  # top n players

        d_players = [p for _, p in frame.iterrows() if p["position"] in constants.D_POSITIONS]
        safety_ids = []
        for index in top_indices:
            safety_ids.append(d_players[index]["nflId"])

        pred_blitzers = self.blitz_model(inputs).numpy().squeeze()
        blitzer_ids = [d_players[i]["nflId"] for i, p in enumerate(pred_blitzers) if p > 0.75]

        pred_man = self.man_model(inputs).numpy().squeeze()
        man_ids = [d_players[i]["nflId"] for i, p in enumerate(pred_man) if p > 0.75]

        if return_confs:
            p_conf = {}
            for i, conf in enumerate(pred_safeties):
                p_conf[d_players[i]["nflId"]] = conf
            return {"coverage": [coverage, coverage_preds.numpy().squeeze()], "safeties": [safety_ids, p_conf], "blitzers": [blitzer_ids, None], "man": [man_ids, None]}
        return {"coverage": coverage, "safeties": safety_ids, "blitzers": blitzer_ids, "man": man_ids}
