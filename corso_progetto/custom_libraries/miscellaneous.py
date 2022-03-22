import sys
from math import floor


def progress_bar(executed, max=100, pre="", post=""):
    sys.stdout.write('\r')
    sys.stdout.write(
        (str(pre) + "[%-20s] %d%%" + str(post)) % ('=' * floor(executed / max * 20), floor(100 * executed / max)))
    sys.stdout.flush()
