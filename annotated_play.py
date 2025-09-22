from collections import defaultdict

from data_compiler import DataCompiler
from play_analyzer import PlayAnalyzer

class AnnotatedPlay:
    """A class to store frame-by-frame predictions and locations of a specific play"""
    def __init__(self, dc: DataCompiler, analyzer: PlayAnalyzer, game_id, play_id):
        """
        Initializes empty arrays to store positions and predictions
        :param dc: the DataCompiler instance
        :param analyzer: the PlayAnalyzer instance for getting predictions from each frame
        :param game_id: the game id of the specified play
        :param play_id: the play id of the specified play
        """
        self.dc = dc
        self.analyzer = analyzer
        self.game_id = game_id
        self.play_id = play_id
        self.play_info = None  # play data to get offensive formation, etc

        # ground truths for coverage and offensive formation
        self.real_coverage = ""
        self.real_off_formation = ""

        # stores frame-by-frame player positions, coverage predictions, confidences, etc
        self.play_frames = []
        self.coverage_frames = []
        self.coverage_conf_frames = []
        self.off_formation_frames = []
        self.off_formation_conf_frames = []
        self.deep_safety_frames = []
        self.safety_conf_frames = []
        self.final_deep_safeties = []
        self.blitzer_frames = []
        self.man_frames = []

        self.before_snap_len = 0

        self._load_data()  # read in data from dc into arrays defined above

    def _load_data(self):
        """Reads in play data and gets coverage, formation, and player predictions for each frame"""
        play_frames = self.dc.get_play_tracking_by_id(game_id=self.game_id, play_id=self.play_id)
        self.play_frames = play_frames

        info = self.dc.get_play_data_by_id(self.game_id, self.play_id)
        self.play_info = info
        self.real_coverage = info["pff_passCoverage"].iloc[0]
        self.real_off_formation = info["offenseFormation"].iloc[0]

        for frame in play_frames:
            if frame.iloc[0]["frameType"] == "BEFORE_SNAP":
                predictions = self.analyzer.analyze_frame(frame, return_confs=True)
                self.coverage_frames.append(predictions["coverage"][0])
                self.coverage_conf_frames.append(predictions["coverage"][1])
                self.safety_conf_frames.append(predictions["safeties"][1])
                self.off_formation_conf_frames.append(predictions["off_formation"][1])
                self.off_formation_frames.append(predictions["off_formation"][0])
                safeties = predictions["safeties"][0]
                self.deep_safety_frames.append(safeties)
                blitzers = [p for p in predictions["blitzers"][0] if p not in safeties]
                self.blitzer_frames.append(blitzers)
                man = [p for p in predictions["man"][0] if p not in safeties]
                self.man_frames.append(man)
            else:
                self.before_snap_len = len(self.coverage_frames)
                break

    def _postprocess_data(self):
        """Uses average confidence to find two final safeties based on all frame results"""
        player_scores = defaultdict(float)
        frame_count = 0

        for p_conf in self.safety_conf_frames:
            frame_count += 1
            for pid, prob in p_conf.items():
                player_scores[pid] += prob

        for pid in player_scores:
            player_scores[pid] /= frame_count

        total_safeties = 0
        for s in self.deep_safety_frames:
            total_safeties += len(s)
        num_safeties = round(total_safeties / frame_count)  # avg to find number of safeties
        best_ids = sorted(player_scores, key=player_scores.get, reverse=True)[:num_safeties]
        self.final_deep_safeties = best_ids

    def __len__(self):
        """Returns the number of frames"""
        return len(self.play_frames)

    def __getitem__(self, key):
        """
        Returns info from a frame at the specified index.
        If the index is after the snap, predictions are from the moment of the snap
        :param key: the index of the frame
        :return: a dictionary containing player data and predictions, and overall predictions
        """
        if isinstance(key, int):
            idx = key
            if key >= self.before_snap_len:
                idx = self.before_snap_len - 1
            return {"players": self.play_frames[key],
                    "coverage": self.coverage_frames[idx],
                    "deep_safeties": self.deep_safety_frames[idx],
                    "deep_safeties_conf": self.safety_conf_frames[idx],
                    "blitzers": self.blitzer_frames[idx],
                    "man": self.man_frames[idx],
                    "off_formation": self.off_formation_frames[idx]}
        else:
            raise TypeError(f"Invalid index type (given {type(key)}).")