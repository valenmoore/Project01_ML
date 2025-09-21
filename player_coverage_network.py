from tensorflow.keras import Model, layers, Sequential
import tensorflow as tf

class PlayerCoverageNetwork(Model):
    def __init__(self, **kwargs):
        super(PlayerCoverageNetwork, self).__init__(**kwargs)  # initialize Model class

        self.player_features = layers.TimeDistributed(
            # loop through each player
            layers.Dense(12, activation='relu')  # 12 features for each player
        )

        self.hidden = Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
        ])
        self.classifier = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # inputs: (B, P, F)
        B, P, F = tf.unstack(tf.shape(inputs))
        features = self.player_features(inputs)  # (B, P, F')
        x = self.hidden(features)  # (B, P, H)
        x = self.classifier(x)
        # x = tf.squeeze(x, axis=-1)
        return x