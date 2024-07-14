import time


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        pass


    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self,unit="s"):
        """停止计时器并将时间记录在列表中"""
        # self.times.append()
        if unit=='s':#单位为秒
            return time.time()-self.tik
        else : #单位为分钟
            return int(time.time() - self.tik) / 60
