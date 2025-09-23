import os
import pandas as pd
import pickle

class DataCompiler:
    """A class to load in data from NFL Big Data Bowl data files into Pandas DataFrames"""

    def __init__(self):
        """Loads data from NFL Big Data Bowl data files"""
        self.data_path = "parquet_data/week_5_data.parquet"
        self.tracking_dir_path = "nfl-big-data-bowl-2025/old_tracking"
        self.players_data_path = "./nfl-big-data-bowl-2025/players.csv"
        self.plays_data_path = "./nfl-big-data-bowl-2025/plays.csv"
        self.player_play_data_path = "./nfl-big-data-bowl-2025/player_play.csv"

        self.data = self.load_data()
        self.players = self.load_player_data()
        self.data = self.merge_add_position_to_data()
        self.plays_data = self.load_plays_data()
        self.player_play_data = self.load_player_play_data()
        self.game_ids = self.data['gameId'].unique()
        self.play_ids = self.data['playId'].unique()

    def merge_add_position_to_data(self):
        """Merge self.data with self.players DataFrame on 'nflId'"""
        return self.data.merge(self.players[['nflId', 'position']], on='nflId', how='left')

    def load_data(self):
        """
        Loads tracking data from a specified number of csv files
        :return: a Pandas DataFrame of the tracking data (unordered)
        """

        try:
            # try to load pre-saved data so we don't have to look at CSVs
            data = pd.read_parquet(self.data_path, engine="fastparquet")
            print("Loaded pre-saved data from", self.data_path)
            return data
        except Exception as e:
            # parquet file is empty, so we have to read from CSVs
            print(e)
            df_list = []
            MIN_COUNT = 4  # first file to read
            MAX_COUNT = 5  # last file to read
            files = sorted(os.listdir(self.tracking_dir_path))
            for i, filename in enumerate(files):
                if filename.endswith(".csv") and MIN_COUNT <= i < MAX_COUNT:
                    print("Loading", filename)
                    df = pd.read_csv(f'{self.tracking_dir_path}/{filename}', dtype={'time': str})
                    df_list.append(df)
            data = pd.concat(df_list, ignore_index=True)

            data.to_parquet(self.data_path, engine='fastparquet', compression='snappy')
            return data

    def load_player_data(self):
        """Loads player data from players.csv"""
        df = pd.read_csv(self.players_data_path)
        return df

    def load_plays_data(self):
        """Loads plays data from plays.csv"""
        df = pd.read_csv(self.plays_data_path)
        return df

    def load_player_play_data(self):
        """Loads player data from each play player_play.csv"""
        df = pd.read_csv(self.player_play_data_path)
        return df

    def get_play_data_by_id(self, game_id, play_id):
        """
        Fetches play information by game_id and play_id
        :param game_id: the game id
        :param play_id: the play id
        :return: a Pandas DataFrame of the play data
        """
        return self.plays_data[(self.plays_data['gameId'] == game_id) & (self.plays_data['playId'] == play_id)]

    def get_play_ids_from_game_id(self, gid):
        """
        Gets plays from a given game
        :param gid: the game id
        :return: an array of play ids
        """
        return self.data[self.data['gameId'] == gid]['playId'].unique()

    def get_player_play_data(self, game_id, play_id, player_id):
        """
        Fetches player data by game_id and play_id
        :param game_id: game id
        :param play_id: play id
        :param player_id: the player id
        :return: a Pandas DataFrame of the matching player play data
        """
        return self.player_play_data[
            (self.player_play_data['gameId'] == game_id) &
            (self.player_play_data['playId'] == play_id) &
            (self.player_play_data['nflId'] == player_id)]

    def get_all_player_play_data(self, game_id, play_id):
        """
        Fetches player data for all players by game_id and play_id
        :param game_id: the game id
        :param play_id: the play id
        :return: a Pandas DataFrame of player play data for all players
        """
        return self.player_play_data[
            (self.player_play_data['gameId'] == game_id) &
            (self.player_play_data['playId'] == play_id)]

    def get_play_tracking_by_id(self, game_id, play_id):
        """
        Fetches tracking data for all players by game_id and play_id
        :param game_id: the game id
        :param play_id: the play id
        :return: a Pandas DataFrame of tracking data for all players, with each row being player data at a frame
        """
        # filter the DataFrame directly by gameId and playId
        play_data = self.data[(self.data['gameId'] == game_id) & (self.data['playId'] == play_id)]
        # group by each timestamp and collect frames as lists
        frame_list = [frame_df for _, frame_df in play_data.groupby("frameId", sort=False)]
        return frame_list

    def get_play_ids_by_coverage(self, coverage_name):
        """
        Gets all plays that match a certain defensive coverage
        :param coverage_name: the defensive coverage name
        :return: a Pandas Dataframe of pairs: (game id, play id) for all plays with the specified coverage
        """
        filtered = self.plays_data[self.plays_data["pff_passCoverage"] == coverage_name]
        # make sure all filtered are in tracking data
        valid_pairs = self.data[['gameId', 'playId']].drop_duplicates()
        merged = filtered.merge(valid_pairs, on=['gameId', 'playId'], how='inner')

        return merged[['gameId', 'playId']]

    def get_all_plays_movement(self, position_filter=None, before_snap=False):
        """
        Gets all tracking data from each play, each frame grouped together containing all players
        :param position_filter: the positions to include in the tracking data (defaults to None; includes all players)
        :param before_snap: if True, only returns tracking data from before the snap
        :return: a Pandas DataFrame of tracking data, grouped by play and then frame
        """
        print("Starting all plays movement")
        df = self.data
        if position_filter is not None:
            df = df[df['position'].isin(position_filter)]
        if before_snap:
            df = df[df['frameType'] == "BEFORE_SNAP"]

        movement_list = []

        for (game_id, play_id), play_df in df.groupby(['gameId', 'playId'], sort=False):
            frames = [frame_df for _, frame_df in play_df.groupby('frameId', sort=False)]
            movement_list.append(frames)
        print("Finished all plays movement")
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
