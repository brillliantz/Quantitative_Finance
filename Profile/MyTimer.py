import time as tm
import itertools
import numpy as np
class MyTimerCLS(object):
    def __init__(self):
        self.start_time = tm.time()
        self._time_list = [0.]
        self.interval_list = []
        self.event_list = []
    def refresh(self, event):
        current_time = tm.time() - self.start_time
        self._time_list.append(current_time)
        self.event_list.append(event)
    """ len(self._time_list) is 1 longer than len(self.event_list)""" 
    def show(self):
        import itertools
        self._time_list = np.array(self._time_list)
        self.interval_list = self._time_list[1:] - self._time_list[:-1]
        for t, e in itertools.izip(self.interval_list, self.event_list):
            print "{0:.4f} s for: {1}.".format(t, e)
        print "{0:.4f} s in total".format(self._time_list[-1])