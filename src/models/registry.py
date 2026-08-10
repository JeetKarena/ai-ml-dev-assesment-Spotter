"""Model registry placeholder."""


class ModelRegistry:
    """Simple registry for model classes."""

    def __init__(self):
        self._models = {}

    def register(self, name, model):
        self._models[name] = model
        return model

    def get(self, name):
        return self._models.get(name)
