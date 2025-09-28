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

        self.defense_features = layers.TimeDistributed(
            # loop through each player
            layers.Dense(12, activation='relu')  # 12 features for each player
        )

        # same as above but for offense
        self.offense_features = layers.TimeDistributed(
            layers.Dense(12, activation='relu')
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
        def_inputs, off_inputs = inputs
        def_features = self.defense_features(def_inputs)  # generate feature vector for each player
        off_features = self.offense_features(off_inputs)  # feature vector for each o player
        off_context = tf.reduce_mean(off_features, axis=1)  # flatten into offense vector

        off_context = tf.expand_dims(off_context, 1)  # (B, 1, 12)
        off_context = tf.repeat(off_context, tf.shape(def_features)[1], axis=1)  # repeat to stack on each defensive player

        def_features = tf.concat([def_features, off_context], axis=-1)  # add offensive context to each d player
        x = self.hidden(def_features)  # reduce features with dense layers for each player
        x = self.classifier(x)  # one binary output for each player
        # x = tf.squeeze(x, axis=-1)
        return x