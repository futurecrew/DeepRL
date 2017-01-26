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
        
def parse_log(log_file):        
    for one_line in open(log_file, 'r'):
        if one_line.startswith('[ Test '):
            data = one_line.split('avg score: ')[1]
            print data.split(' elapsed')[0]        
            
if __name__ == '__main__':
    log_file = 'output/hero_20170121_104059.log'
    parse_log(log_file)
    
