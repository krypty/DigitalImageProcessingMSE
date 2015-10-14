#!/bin/python3
# coding: utf-8

import datetime
import math
import time


def timeit(method):

    def timed(*args, **kw):
        t0 = time.time()
        result = method(*args, **kw)
        t1 = time.time()

        execution_time = round((t1 - t0) * 1000, 3)
        print("[timeit] %s() finished in %s ms" % (method.__name__, execution_time))
        return result

    return timed


if __name__ == '__main__':

    @timeit
    def s(seconds):
        time.sleep(seconds)

    s(seconds=1)
    s(seconds=2)
    s(seconds=2.5)
