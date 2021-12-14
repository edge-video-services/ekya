import csv
import os
import time

from ekya import CONFIG


class Datastream(object):
    def __init__(self, stream_id, max_len=1000, max_time=30, init_list=None, base_dir=None):
        '''

        :param stream_id:
        :param max_len: num of rows after which to flush
        :param max_time: time after which to flush to disk
        :param init_list:
        :param base_dir:
        '''
        self.stream_id = str(stream_id)
        if not init_list:
            init_list = []

        file_path = os.path.join(base_dir, stream_id + '.csv')

        self.data_list = init_list
        self.file_path = file_path
        self.max_len = max_len
        self.last_save_time = time.time()
        self.max_time = max_time  # If it has been more than this time since the last save, save.


    def flush_to_disk(self):
        with open(self.file_path, "a+") as f:
            writer = csv.writer(f)
            writer.writerows(self.data_list)
        self.last_save_time = time.time()
        self.data_list = []

    def append(self, item):
        self.data_list.append(item)
        if (len(self.data_list) > self.max_len) or (time.time() - self.last_save_time > self.max_time):
            self.flush_to_disk()

    def close(self):
        self.flush_to_disk()

class BaseLogger(object):
    def __init__(self, *args, **kwargs):
        pass

    def append(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass

class Logger(BaseLogger):
    def __init__(self, stream_max_len=1000, stream_max_time=30, base_dir=None):
        self.stream_max_time = stream_max_time
        self.stream_max_len = stream_max_len

        if not base_dir:
            base_dir = CONFIG.DEFAULT_LOGGING_PATH
            base_dir = self.setup_dirs(base_dir)
        print("Logging to {}. Max time: {}. Max len: {}".format(base_dir, stream_max_time, stream_max_len))

        self.base_dir = base_dir
        self.datastreams = {}

    def setup_dirs(self, base_dir):
        # Create another folder if the results folder already exists.
        idx = 0
        while True:
            try_base_dir = base_dir + str(idx)
            if not os.path.isdir(try_base_dir):
                os.makedirs(try_base_dir, exist_ok=True)
                return try_base_dir
            idx+=1

    def append(self, stream_id, item):
        if not isinstance(item, list):
            item = [item]
        if stream_id in self.datastreams:
            self.datastreams[stream_id].append(item)
        else:
            self.datastreams[stream_id] = Datastream(stream_id, init_list=[item], max_len=self.stream_max_len,
                                                     max_time=self.stream_max_time, base_dir=self.base_dir)

    def flush(self):
        for stream_id, stream in self.datastreams.items():
            stream.close()

    def __del__(self):
        if hasattr(self, "datastreams"):
            self.flush()