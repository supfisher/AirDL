class ListFn:
    """
        It will serially process the given list function
    """
    def __init__(self, list_fn):
        self.list_fn = list_fn

    def __call__(self, *args, **kwargs):
        for fn in self.list_fn:
            fn(*args, **kwargs)


class ObjectParallel:
    def __init__(self, objects=None):
        self.objects = objects

        for key in dir(objects[0]):
            if '__' not in key:
                setattr(self, key, ListFn([getattr(obj, key)
                                                for obj in self.objects]))
