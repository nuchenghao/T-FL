import torch


class Net():
    def __init__(self, net, trainConfigJSON, init_weights):
        self.net = net
        self.trainConfigJSON = trainConfigJSON
        self.init_weights = init_weights

    def initNet(self):
        self.net.apply(self.init_weights)

    def getNetParams(self):
        return self.net.cpu().state_dict()  # 将模型移到CPU，然后导出模型参数

    def saveNetToFile(self, path):
        torch.save(self.net.state_dict(), path)  # 保存文件参数到path中

    def getNetFromFile(self, path):
        self.net.load_state_dict(torch.load(path))  # 直接读取文件中存储的参数
