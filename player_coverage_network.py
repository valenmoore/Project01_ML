from tensorflow.keras import Model, layers, Sequential
import tensorflow as tf

class PlayerCoverageNetwork(Model):
    """The Tensorflow model used to predict player-by-player defensive assignments like safeties and man defenders"""
    def __init__(self, **kwargs):
        """
        Initializes layers for the PlayerCoverageNetwork class
        :param kwargs: any other arguments from the tf.keras Model class
        """
        super(PlayerCoverageNetwork, self).__init__(**kwargs)  # if mama ain't happy nobody's happy

        self.player_features = layers.TimeDistributed(
            # loop through each player
            layers.Dense(12, activation='relu')  # 12 features for each player
        )

        # a sequence of dense layers connecting player features to the final classifier
        self.hidden = Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
        ])

        # final binary classifier
        self.classifier = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        """
        An override method that calls model layers on inputs
        :param inputs: the inputs to the model (shape (B, 11, 17))
        :param training: whether the model is training or not
        :return: the model output (11, num_classes)
        """
        B, P, F = tf.unstack(tf.shape(inputs))
        features = self.player_features(inputs)  # generate feature vector for each player
        x = self.hidden(features)  # reduce features with dense layers for each player
        x = self.classifier(x)  # one binary output for each player
        # x = tf.squeeze(x, axis=-1)
        return x