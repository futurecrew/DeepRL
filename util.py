import sys
import time

class Logger(object):
    def __init__(self, folder, gameName):
        filename="%s/%s_%s.log" % (folder, gameName, time.strftime('%Y%m%d_%H%M%S'))
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()