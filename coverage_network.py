from tensorflow.keras import Model, layers, Sequential

class CoverageNetwork(Model):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)

        self.player_feature = layers.TimeDistributed(
            layers.Dense(12, activation='relu'),
        )

        self.hidden = Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
        ])

        self.num_classes = num_classes
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # inputs: (B, T, P, F)
        x = inputs
        player_feat = self.player_feature(x)
        x = layers.Flatten()(player_feat)
        x = self.hidden(x)
        out = self.classifier(x)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
