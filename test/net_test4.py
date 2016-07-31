import caffe
import numpy as np
import random

class NetTester():
    def __init__(self):
        
        self.trainBatchSize = 1
        
        caffe.set_mode_gpu()
        
        self.solver = caffe.SGDSolver('models/test/solver_test4.prototxt')
        self.trainNet = self.solver.net
        self.replayMemory = []            
        self.running = True

    def addData(self, frame1, action, reward, frame2, gameOver, stepNo):
        self.replayMemory.append((frame1, action, reward, frame2, gameOver, stepNo))
            
    def gogo(self):
        data1 = np.zeros((84, 84))
        data1[0, 0] = 6
        data1[2, 2] = 1
        data1[8, 15] = 3
        
        data2 = np.zeros((84, 84))
        data2[1, 2] = 5
        data2[19, 71] = 4
        data2[8, 15] = 3
        
        data3 = np.zeros((84, 84))
        data3[1, 2] = 7
        data3[19, 71] = 125
        data3[8, 15] = 7
        
        data4 = np.zeros((84, 84))
        data4[10, 2] = 5
        data4[29, 9] = 4
        data4[8, 1] = 85
        
        self.addData(data1, 1, 0, data2, False, 1)
        self.addData(data1, 2, 0, data3, False, 1)
        self.addData(data2, 2, 1, data3, False, 2)
        self.addData(data3, 2, 0, data4, True, 3)
        self.addData(data3, 1, 0, data4, True, 3)
        
        self.train()
        
    def train(self):
        while self.running:
            trainState = np.zeros((1, 1, 84, 84), dtype=np.float32)
            trainNewState = np.zeros((1, 1, 84, 84), dtype=np.float32)
            
            stateHistoryStack, trainAction, trainReward, newStateHistoryStack, trainGameOver, episodeStep \
                = self.replayMemory[random.randint(0, len(self.replayMemory)-1)]
                    
            trainState[0, :, :, :] = stateHistoryStack
            trainNewState[0, :, :, :] = newStateHistoryStack

            label = np.zeros((1, 4), dtype=np.float32)
                
            self.solver.net.forward(data=trainNewState.astype(np.float32, copy=False),
                                                      labels=label)
            newActionValues = self.solver.net.blobs['cls_score'].data.copy()
                
            self.solver.net.forward(data=trainState.astype(np.float32, copy=False),
                                                      labels=label)
            label = self.solver.net.blobs['cls_score'].data.copy()
            
            if trainGameOver:
                label[0, trainAction] = trainReward
            else:
                label[0, trainAction] = trainReward + 0.99 * np.max(newActionValues[0])

            self.solver.net.blobs['data'].data[...] = trainState.astype(np.float32, copy=False)
            self.solver.net.blobs['labels'].data[...] = label
        
            self.solver.step(1)

            self.solver.net.forward(data=trainState.astype(np.float32, copy=False),
                                                      labels=label)
            classScore2 = self.solver.net.blobs['cls_score'].data.copy()

            print 'step   : %s' % episodeStep
            print 'label  : %s' % label
            print 'score : %s' % classScore2

                            
        print 'ModelRunner thread finished.'

    def finishTrain(self):
        self.running = False
        
if __name__ == '__main__':
    NetTester().gogo()
        