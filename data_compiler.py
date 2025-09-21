import os
import pandas as pd
import pickle

class DataCompiler:
    def __init__(self):
        self.data_path = "parquet_data/week_5_data.parquet"
        self.data = self.load_data()
        self.players = self.load_player_data()
        self.data = self.merge_add_position_to_data()
        self.plays_data = self.load_plays_data()
        self.player_play_data = self.load_player_play_data()
        self.game_ids = self.data['gameId'].unique()
        self.play_ids = self.data['playId'].unique()

    def merge_add_position_to_data(self):
        # Merge self.data with self.players DataFrame on 'nflId'
        return self.data.merge(self.players[['nflId', 'position']], on='nflId', how='left')

    def load_data(self):
        try:
            data = pd.read_parquet(self.data_path, engine="fastparquet")
            print("Loaded pre-saved data from", self.data_path)
            return data
        except Exception as e:
            print(e)
            # parquet file is empty, raises exception
            df_list = []
            MIN_COUNT = 4
            MAX_COUNT = 5
            files = sorted(os.listdir("nfl-big-data-bowl-2025/old_tracking"))
            for i, filename in enumerate(files):
                if filename.endswith(".csv") and MIN_COUNT <= i < MAX_COUNT:
                    print("Loading", filename)
                    df = pd.read_csv(f'nfl-big-data-bowl-2025/old_tracking/{filename}', dtype={'time': str})
                    df_list.append(df)
            data = pd.concat(df_list, ignore_index=True)
            # self.save_data_to_pkl("./data/tracking_data.pkl", data)
            data.to_parquet(self.data_path, engine='fastparquet', compression='snappy')
            return data


    def load_player_data(self):
        df = pd.read_csv(f'./nfl-big-data-bowl-2025/players.csv')
        return df

    def load_plays_data(self):
        df = pd.read_csv(f'./nfl-big-data-bowl-2025/plays.csv')
        return df

    def load_player_play_data(self):
        df = pd.read_csv(f'./nfl-big-data-bowl-2025/player_play.csv')
        return df

    def save_data_to_pkl(self, path, data):
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print("Data saved to", path)

    def load_data_from_pkl(self, path):
        # Read back from the Parquet file
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data is None:
            raise Exception("Empty file")
        return data


    def get_play_data_by_id(self, game_id, play_id):
        return self.plays_data[(self.plays_data['gameId'] == game_id) & (self.plays_data['playId'] == play_id)]

    def get_play_ids_from_game_id(self, gid):
        return self.data[self.data['gameId'] == gid]['playId'].unique()

    def get_player_play_data(self, game_id, play_id, player_id):
        return self.player_play_data[
            (self.player_play_data['gameId'] == game_id) &
            (self.player_play_data['playId'] == play_id) &
            (self.player_play_data['nflId'] == player_id)]

    def get_all_player_play_data(self, game_id, play_id):
        return self.player_play_data[
            (self.player_play_data['gameId'] == game_id) &
            (self.player_play_data['playId'] == play_id)]

    def get_play_tracking_by_id(self, game_id, play_id):
        # Filter the DataFrame directly by gameId and playId
        play_data = self.data[(self.data['gameId'] == game_id) & (self.data['playId'] == play_id)]
        # Group by each timestamp and collect frames as lists
        frame_list = [frame_df for _, frame_df in play_data.groupby("frameId", sort=False)]
        return frame_list

    def get_game_ids_by_coverage(self, coverage_name):
        filtered = self.plays_data[self.plays_data["pff_passCoverage"] == coverage_name]
        # make sure all filtered are in tracking data
        valid_pairs = self.data[['gameId', 'playId']].drop_duplicates()
        merged = filtered.merge(valid_pairs, on=['gameId', 'playId'], how='inner')

        return merged[['gameId', 'playId']]

    def get_player_movement(self, player_id, play):
        # Filter out data for the specified player within the given play DataFrame
        player_data = play[play['nflId'] == player_id]
        return player_data.to_dict(orient='records')

    def get_all_plays_movement(self, position_filter=None, before_snap=False):
        print("before")
        df = self.data
        if position_filter is not None:
            df = df[df['position'].isin(position_filter)]
        if before_snap:
            df = df[df['frameType'] == "BEFORE_SNAP"]

        movement_list = []

        for (game_id, play_id), play_df in df.groupby(['gameId', 'playId'], sort=False):
            frames = [frame_df for _, frame_df in play_df.groupby('frameId', sort=False)]
            movement_list.append(frames)
        print("after")
        return movement_list

    def get_all_player_movements(self, position_filter=None, before_snap=False):
        """
        Compile every single individual player's movement from every play.
        :return: A list of movement arrays, where each movement array is a list of a player's rows over the course of the play
        """
        movement_list = []

        # Filter based on position if specified
        filtered_data = self.data if position_filter is None else self.data[self.data['position'].isin(position_filter)]
        if before_snap:
            filtered_data = filtered_data[filtered_data['frameType'] == "BEFORE_SNAP"]  # filter before snap

        # Group by gameId, playId, and nflId for efficient data retrieval
        grouped = filtered_data.groupby(['gameId', 'playId', 'nflId'])

        for (game_id, play_id, player_id), group in grouped:
            movement_list.append(group.to_dict(orient='records'))

        return movement_list

    def get_player_by_id(self, id):
        return self.players[self.players['nflId'] == id]