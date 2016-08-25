import sys

class Logger(object):
    def __init__(self, logFile):
        self.terminal = sys.stdout
        self.log = open(logFile, "w")
        
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()