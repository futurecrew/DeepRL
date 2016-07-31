import caffe
import numpy as np
import random

class NetTester():
    def __init__(self):
        
        self.trainBatchSize = 1
        
        #caffe.set_mode_gpu()
        
        self.solver = caffe.SGDSolver('models/test/solver_test2.prototxt')
        self.trainNet = self.solver.net
        self.replayMemory = []            
        self.running = True

    def addData(self, input, output):
        self.replayMemory.append((input, output))
            
    def gogo(self):
        self.addData([[6, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     [0, 3, 0, 0])
        self.addData([[0, 0, 0, 0, 0, 9, 0, 0, 0, 0], 
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     [0, 0, 0, 2])
        
        self.train()
        
    def train(self):
        self.step = 0
        while self.running:
            for i in range(0, self.trainBatchSize):
                input, output = self.replayMemory[random.randint(0, len(self.replayMemory)-1)]
                    
                if i == 0:
                    trainData = np.zeros((self.trainBatchSize, 1, 10, 10), dtype=np.float32)
                    label = np.zeros((self.trainBatchSize, 4), dtype=np.float32)
                 
                trainData[i, :] = input
                label[i, :] = output

            self.solver.net.forward(data=trainData.astype(np.float32, copy=False),
                                                      labels=label)
            classScore1 = self.solver.net.blobs['cls_score'].data.copy()
            
            self.solver.net.blobs['data'].data[...] = trainData.astype(np.float32, copy=False)
            self.solver.net.blobs['labels'].data[...] = label
    
            self.solver.step(1)

            self.solver.net.forward(data=trainData.astype(np.float32, copy=False),
                                                      labels=label)
            classScore2 = self.solver.net.blobs['cls_score'].data.copy()

            self.step += 1
            print label
            print classScore2
            
        print 'ModelRunner thread finished.'

    def finishTrain(self):
        self.running = False
        
if __name__ == '__main__':
    NetTester().gogo()
        