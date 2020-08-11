import sys


class Logger(object):
    """
        sys.stdout=Logger("test.txt")
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        pass
