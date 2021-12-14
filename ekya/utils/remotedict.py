import ray

class Dict(object):
    def __init__(self):
        self.d = {} # id -> handle map

    def put(self, key, value):
        self.d[key] = value
        return True

    def get(self, key):
        return self.d[key]

RemoteDict = ray.remote(Dict)