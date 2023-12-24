class messageInServer:
    def __init__(self, net, totalEpoch, totalClient, numLocalTrain, batchSize, learningRate, splitDataSet):
        self.register = False  # 先进行注册，然后开始训练；判断是否所有client注册完成


        # server状态
        self.net = net
        self.currentEpoch = 0
        self.totalEpoch = totalEpoch
        self.currentClient = 0
        self.totalClient = totalClient

        # 需要传递给client的参数
        self.numLocalTrain = numLocalTrain
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.splitDataset = splitDataSet

    def addEpoch(self):
        self.currentEpoch += 1

    def addClient(self):
        self.currentClient += 1

    def clearClient(self):
        self.currentClient = 0

    def ready(self):
        if self.currentClient == self.totalClient:
            return True
        else:
            return False

    def finish(self):
        if self.currentEpoch == self.totalEpoch:
            return True
        else:
            return False
