import time


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        pass
        """Defined in :numref:`subsec_linear_model`"""
        # self.times = []
        # self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        # self.times.append()
        return int(time.time() - self.tik) / 60
