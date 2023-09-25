# This script contain decorators help developing.
# Author:
#     Albert Dongz
# History:
#     2020.4.17 First Release
# Dependencies:
#     time
# Attention:
#     1. Nothing

import time


def exec_time(func):
    """
    This function is a wrapper which can measure the exec time.

    Arguments:
        func {function} -- func need to be measure.

    Returns:
        function -- a func can measure exec time.
    """
    def new_func(*args, **args2):
        t0 = time.time()
        print("@%s, {%s} start" %
              (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("@%s, {%s} end" %
              (time.strftime("%X", time.localtime()), func.__name__))
        print("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back

    return new_func
