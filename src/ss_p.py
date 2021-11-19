

class Preprocessing:
    def __init__(self) -> None:
        pass

    def normalization(self, input, bounds: dict):
        return (input - bounds['Lower']) / (bounds['Upper'] - bounds['Lower'])

    def denormalization(self, input, type, bounds: dict):
        if type == 'integer' or type == 'categorical':
            return round(bounds['Lower'] + input * (bounds['Upper'] - bounds['Lower']))
        elif type == 'float':
            return bounds['Lower'] + input * (bounds['Upper'] - bounds['Lower'])
