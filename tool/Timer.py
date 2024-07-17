import time


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        pass

    def start(self):
        self.tik = time.time()

    def stop(self, unit="s"):
        if unit == 's':  # 单位为秒
            return int(time.time() - self.tik)
        else:  # 单位为分钟
            return int(time.time() - self.tik) / 60
