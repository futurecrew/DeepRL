logFile = 'output/space_invaders_20160901_204149.log'

for oneLine in open(logFile, 'r'):
    if oneLine.startswith('[ Test '):
        data = oneLine.split('avg score: ')[1]
        print data.split('. elapsed')[0]