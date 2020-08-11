import time


class Timer(object):
    def __init__(self):
        self.time_begin = time.time()  # 计时点
        self.time_consume = 0  # 累计时间

    def print_time_info(self):
        time_epoch = time.time() - self.time_begin
        self.time_consume += time_epoch
        self.time_begin = time.time()
        info_display = "* Epoch Time = " + str(int(time_epoch)) + "s[" + str(int(time_epoch / 60)) + " min]     " + \
                       "ALL Time = " + str(int(self.time_consume)) + "s[" + str(int(self.time_consume / 60)) + " min]" \
                       + "\n\n"
        # 单位是秒
        print(info_display)

    def get_time_info(self):
        time_epoch = time.time() - self.time_begin
        self.time_consume += time_epoch
        self.time_begin = time.time()
        return time_epoch, self.time_consume


if __name__ == '__main__':
    t = Timer()
    time.sleep(1.5)
    t.print_time_info()

    time.sleep(2.5)
    t.print_time_info()
