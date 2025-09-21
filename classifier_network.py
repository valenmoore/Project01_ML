import keras.src.saving
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
import constants
from data_compiler import DataCompiler
import tensorflow as tf
from tensorflow.keras import layers, models

class Classifier:
    def __init__(self, dc: DataCompiler):
        self.dc = dc
        self.man_zone_model = self.load_model("./models/man_zone_model.keras")
        self.offense_formation_model = self.load_model("./models/offense_formation.keras")
        self.coverage_model = self.load_model("./models/coverage_model.keras")

    def _route_to_class(self, route):
        if not route in constants.ROUTES:
            print("NEED TO APPEND: ", route)
            constants.ROUTES.append(route)
        return constants.ROUTES.index(route)

    def train_man_zone_classifier(self):
        movements = self.dc.get_all_plays_movement(position_filter=constants.D_POSITIONS, before_snap=True)
        wanted_keys = ['x', 'y']  # values to pass into network
        X, y = [], []
        for play in movements:
            play_id = play[0][0]['playId']
            game_id = play[0][0]['gameId']
            play_data = self.dc.get_play_data_by_id(game_id, play_id)
            man_zone_str = play_data['pff_manZone'].iloc[0]
            if man_zone_str in constants.MAN_ZONE:
                man_zone_label = constants.MAN_ZONE.index(man_zone_str)
                for frame in play:
                    features = []
                    for player in frame:
                        features.append([player.get(key, 0) for key in wanted_keys])
                    features = np.array(features)
                    flat = features.flatten()
                    if len(flat) > 22:
                        print("BAD AT", game_id, play_id)
                        for p in frame:
                            print(p['position'])
                    X.append(flat)
                    y.append(man_zone_label)
        print("Completed constructing dataset.")

        # Convert X and y to numpy arrays (X needs to be 2D array)
        X = np.array(X)  # Array of objects to handle variable-length lists
        y = np.array(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = self.create_binary_model((22,))
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        self.save_model(model, "./models/man_zone_model.keras")
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")

    def create_binary_model(self, input_shape):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def save_model(self, model, path):
        print("Model saved to ", path)
        model.save(path)

    def load_model(self, path):
        return keras.src.saving.load_model(path)

    def predict_man_zone(self, player_rows):
        d_poses = [[p['x'], p['y']] for p in player_rows if p['position'] in constants.D_POSITIONS]
        d_poses = np.array(d_poses).flatten()
        batch = np.array([d_poses])
        prediction = self.man_zone_model.predict(batch, verbose=0)  # turn off print output
        binary_prediction = (prediction > 0.5).astype(int)[0][0]  # Converts probabilities to 0 or 1
        return constants.MAN_ZONE[binary_prediction]

    def predict_formation(self, player_rows):
        o_poses = [[p['x'], p['y']] for p in player_rows if p['position'] in constants.O_POSITIONS]
        o_poses = np.array(o_poses).flatten()
        batch = np.array([o_poses])
        prediction = self.offense_formation_model.predict(batch, verbose=0)  # turn off print output
        predicted_class = np.argmax(prediction)  # Index of the max probability
        return constants.OFFENSE_FORMATIONS[predicted_class]

    def create_multiclass_model(self, input_shape, num_classes):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_offense_formation(self):
        movements = self.dc.get_all_plays_movement(position_filter=constants.O_POSITIONS, before_snap=True)
        wanted_keys = ['x', 'y']  # values to pass into network
        X, y = [], []
        for play in movements:
            play_id = play[0][0]['playId']
            game_id = play[0][0]['gameId']
            play_data = self.dc.get_play_data_by_id(game_id, play_id)
            formation_str = play_data['offenseFormation'].iloc[0]
            if formation_str in constants.OFFENSE_FORMATIONS:
                formation_label = constants.OFFENSE_FORMATIONS.index(formation_str)
                for frame in play:
                    features = []
                    for player in frame:
                        features.append([player.get(key, 0) for key in wanted_keys])
                    features = np.array(features)
                    flat = features.flatten()
                    if len(flat) != 22:
                        print("BAD AT", game_id, play_id)
                        for p in frame:
                            print(p['position'], p['club'], p['displayName'])
                    else:
                        X.append(flat)
                        y.append(formation_label)
        print("Completed constructing dataset.")

        # Convert X and y to numpy arrays (X needs to be 2D array)
        X = np.array(X)  # Array of objects to handle variable-length lists
        y = np.array(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = self.create_multiclass_model((22,), len(constants.OFFENSE_FORMATIONS))
        model.fit(X_train, y_train, epochs=50, batch_size=32)
        self.save_model(model, "./models/offense_formation.keras")
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")

    def train_coverage_classifier(self):
        movements = self.dc.get_all_plays_movement(position_filter=constants.D_POSITIONS, before_snap=True)
        wanted_keys = ['x', 'y']  # values to pass into network
        X, y = [], []
        for play in movements:
            play_id = play[0][0]['playId']
            game_id = play[0][0]['gameId']
            play_data = self.dc.get_play_data_by_id(game_id, play_id)
            coverage_str = play_data['pff_passCoverage'].iloc[0]
            if coverage_str in constants.COVERAGE_FORMATIONS:
                coverage_label = constants.COVERAGE_FORMATIONS.index(coverage_str)
                for frame in play:
                    features = []
                    for player in frame:
                        game_id = player['gameId']
                        features.append([player.get(key, 0) for key in wanted_keys])
                    features = np.array(features)
                    flat = features.flatten()
                    if len(flat) > 22:
                        print("BAD AT", game_id, play_id)
                        print(frame)
                        for p in frame:
                            print(f"{p['displayName']} from {p['club']} incorrectly labeled as defense at {p['position']}, gameid: {p['gameId']} pid: {p['playId']}")
                    else:
                        # if num of features is correct
                        X.append(flat)
                        y.append(coverage_label)
        print("Completed constructing dataset.")

        # Convert X and y to numpy arrays (X needs to be 2D array)
        X = np.array(X)  # Array of objects to handle variable-length lists
        y = np.array(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = self.create_multiclass_model((22,), len(constants.COVERAGE_FORMATIONS))
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
        self.save_model(model, "./models/coverage_model.keras")
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")

    def predict_coverage(self, player_rows):
        d_poses = [[p['x'], p['y']] for p in player_rows if p['position'] in constants.D_POSITIONS]
        d_poses = np.array(d_poses).flatten()
        batch = np.array([d_poses])
        prediction = self.coverage_model.predict(batch, verbose=0)  # turn off print output
        predicted_class = np.argmax(prediction)  # Index of the max probability
        confidence = prediction[0][predicted_class] * 100  # Get confidence as a percentage
        return constants.COVERAGE_FORMATIONS[predicted_class], confidence