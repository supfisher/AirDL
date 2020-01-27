
class Channel:
    """
        data is a tensor
        constraints is a class
    """
    def __init__(self, constraints):
        self.constraints = constraints



class Gaussian(Channel):
    def __init__(self, constraints=None):
        super(Gaussian, self).__init__(constraints)

    def __call__(self, data, *args, **kwargs):
        return data

