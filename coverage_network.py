from tensorflow.keras import Model, layers, Sequential

class CoverageNetwork(Model):
    """The Tensorflow model used to predict overall offensive and defensive schemes"""

    def __init__(self, num_classes=10, **kwargs):
        """
        Initializes layers for the CoverageNetwork class
        :param num_classes: the number of classes for the final classifier
        :param kwargs: any other arguments from the tf.keras Model class
        """
        super().__init__(**kwargs)  # if mama ain't happy nobody's happy

        # iterates through each player and generates a feature vector
        self.player_feature = layers.TimeDistributed(
            layers.Dense(12, activation='relu')
        )

        # a sequence of dense layers connecting player features to the final classifier
        self.hidden = Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
        ])

        # outputs the final classification as a softmax prediction
        self.classifier = layers.Dense(num_classes, activation='softmax')

        self.num_classes = num_classes

    def call(self, inputs, training=False):
        """
        An override method that calls model layers on inputs
        :param inputs: the inputs to the model (shape (B, 11, 17))
        :param training: whether the model is training or not
        :return: the model output (num_classes)
        """
        # inputs: (batch, players, features)
        x = inputs
        player_feat = self.player_feature(x)  # generate feature vector for each player
        x = layers.Flatten()(player_feat)  # flatten feature vectors into one team vector
        x = self.hidden(x)  # reduce features through decreasing dense layers
        out = self.classifier(x)  # final output for whole team
        return out

    def get_config(self):
        """Makes the model load properly with correct number of classes when saved to .keras"""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Makes the model load properly with correct number of classes when saved to .keras"""
        return cls(**config)
