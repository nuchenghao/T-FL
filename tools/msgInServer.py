class messageInServer:
    def __init__(self, totalEpoch):
        # self.finished = False
        self.currentEpoch = 0
        self.totalEpoch = totalEpoch

    def addEpoch(self):
        self.currentEpoch += 1

    def finish(self):
        if self.currentEpoch == self.totalEpoch:
            return True
        else:
            return False
