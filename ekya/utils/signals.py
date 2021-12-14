import ray.experimental.signal as signal

class StopInferenceLoopSignal(signal.Signal):
    def __init__(self, target_id):
        self.target_id = target_id

class IsAliveSignal(signal.Signal):
    def __init__(self, model_id):
        self.model_id = model_id

class UserSignal(signal.Signal):
    def __init__(self, value):
          self.value = value

    def get_value(self):
          return self.value

