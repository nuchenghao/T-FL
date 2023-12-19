class messageInClient:
    def __init__(self, net):
        self.finished = False
        self.net = net
        self.numLocalTrain = 0
        self.batchSize = 0
        self.learningRate = 0
