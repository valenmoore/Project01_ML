from tensorflow.keras.models import load_model
import numpy as np

import constants
from coverage_network import CoverageNetwork
from player_coverage_network import PlayerCoverageNetwork
from offense_aware_player_coverage import PlayerCoverageNetwork as OffenseAwarePlayerCoverageNetwork

class PlayAnalyzer:
    """Uses saved models to get predictions for a play"""

    def __init__(self):
        """Loads models for predictions"""
        self.coverage_model_path = "./models/player_coverage/19/model.keras"
        self.off_formation_model_path = "./models/offense_formation/0/model.keras"
        self.coverage_model = self._load_coverage_model()
        self.player_models = self._load_player_models()
        self.off_formation_model = self._load_off_formation_model()
        self.blitz_models = self._load_blitz_models()
        self.man_models = self._load_man_models()

    def _load_off_formation_model(self):
        return load_model(self.off_formation_model_path, custom_objects={"CoverageNetwork": CoverageNetwork})

    def _load_coverage_model(self):
        return load_model(self.coverage_model_path, custom_objects={"GNNCoverageNetwork": CoverageNetwork})

    def _load_player_models(self):
        """Loads safety models for each type of coverage, stored in a dictionary by coverage name"""
        coverages = ["Cover-1", "Cover-2", "Cover-3", "Quarters"]
        models = {}
        for coverage in coverages:
            path = f"./models/{coverage}_player/best/model.keras"
            models[coverage] = load_model(path, custom_objects={"PlayerCoverageNetwork": PlayerCoverageNetwork})
        return models

    def _load_blitz_models(self):
        """Loads man models for each type of coverage, stored in a dictionary by coverage name"""
        coverages = ["Cover-1", "Cover-2", "Cover-3", "Quarters"]
        models = {}
        for coverage in coverages:
            path = f"./models/blitz_models/{coverage}.keras"
            models[coverage] = load_model(path, custom_objects={"PlayerCoverageNetwork": OffenseAwarePlayerCoverageNetwork})
        return models

    def _load_man_models(self):
        """Loads man models for each type of coverage, stored in a dictionary by coverage name"""
        coverages = ["Cover-1", "Cover-2", "Cover-3", "Quarters"]
        models = {}
        for coverage in coverages:
            path = f"./models/man_models/{coverage}.keras"
            models[coverage] = load_model(path, custom_objects={"PlayerCoverageNetwork": OffenseAwarePlayerCoverageNetwork})
        return models

    def _construct_model_inputs(self, frame, is_off_aware):
        """Takes tracking data from a frame and constructs the input tensor that all models expect"""
        wanted_keys = ['x', 'y', 's', 'a', 'o', 'dir']  # values to pass into network

        model_inputs_d = []
        model_inputs_o = []
        for _, player in frame.iterrows():
            # if not offense aware only look at defensive players
            if player["position"] in constants.D_POSITIONS:
                pos_idx = constants.D_POSITIONS.index(player['position'])
                one_hot = np.zeros(len(constants.D_POSITIONS))
                one_hot[pos_idx] = 1

                features = player[wanted_keys].to_numpy()

                features = np.hstack([features, one_hot])
                model_inputs_d.append(features)
            if is_off_aware and player["position"] in constants.O_POSITIONS:
                pos_idx = constants.O_POSITIONS.index(player['position'])
                one_hot = np.zeros(len(constants.O_POSITIONS))
                one_hot[pos_idx] = 1

                features = player[wanted_keys].to_numpy()

                features = np.hstack([features, one_hot])
                model_inputs_o.append(features)

        d_inputs = np.array(model_inputs_d, dtype=np.float32)
        d_inputs = np.expand_dims(d_inputs, axis=0)
        if is_off_aware:
            o_inputs = np.array(model_inputs_o, dtype=np.float32)
            o_inputs = np.expand_dims(o_inputs, axis=0)
            return d_inputs, o_inputs

        return d_inputs

    def analyze_frame(self, frame, return_confs=False):
        d_inputs, o_inputs = self._construct_model_inputs(frame, is_off_aware=True)
        coverage_preds = self.coverage_model(d_inputs)
        coverage_pred = np.argmax(coverage_preds)
        coverage = constants.COMMON_COVERAGE_FORMATIONS[coverage_pred]

        formation_preds = self.off_formation_model(d_inputs)
        formation_pred = np.argmax(formation_preds)
        formation = constants.OFFENSE_FORMATIONS[formation_pred]

        num_players_map = {
            "Cover-1": 1,
            "Cover-2": 2,
            "Cover-3": 3,
            "Quarters": 4
        }

        num_players = num_players_map[coverage]
        pred_safeties = self.player_models[coverage](d_inputs).numpy().squeeze()
        top_indices = np.argsort(pred_safeties)[-num_players:]  # top n players

        d_players = [p for _, p in frame.iterrows() if p["position"] in constants.D_POSITIONS]
        safety_ids = []
        for index in top_indices:
            safety_ids.append(d_players[index]["nflId"])

        blitz_model = self.blitz_models[coverage]
        pred_blitzers = blitz_model((d_inputs, o_inputs)).numpy().squeeze()
        blitzer_ids = [d_players[i]["nflId"] for i, p in enumerate(pred_blitzers) if p > 0.5]

        man_model = self.man_models[coverage]
        pred_man = man_model((d_inputs, o_inputs)).numpy().squeeze()
        man_ids = [d_players[i]["nflId"] for i, p in enumerate(pred_man) if p > 0.5]

        if return_confs:
            p_conf = {}
            for i, conf in enumerate(pred_safeties):
                p_conf[d_players[i]["nflId"]] = conf
            return {"coverage": [coverage, coverage_preds.numpy().squeeze()],
                    "off_formation": [formation, formation_preds.numpy().squeeze()],
                    "safeties": [safety_ids, p_conf],
                    "blitzers": [blitzer_ids, None],
                    "man": [man_ids, None]}
        return {"coverage": coverage,
                "off_formation": formation,
                "safeties": safety_ids,
                "blitzers": blitzer_ids,
                "man": man_ids}
