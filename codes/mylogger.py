# -*- coding: utf-8 -*-
import logging  # 引入logging模块
import os
import time

def get_mylogger(log_path):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关

    # 第二步，创建一个handler，用于写入日志文件
    os.makedirs(os.path.join(log_path,time.strftime('%Y-%m-%d', time.localtime(time.time()))),exist_ok=True)
    log_path = os.path.join(log_path,time.strftime('%Y-%m-%d', time.localtime(time.time())))
    rq = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_name = os.path.join(log_path,rq+'.log')
    logfile = log_name
    print(log_name)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)  # 输出到file的log等级的开关

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 输出到console的log等级的开关

    # 第三步，定义handler的输出格式
    fh_formatter = logging.Formatter("%(filename)s : %(message)s")
    fh.setFormatter(fh_formatter)
    ch_formatter = logging.Formatter("[line:%(lineno)d] :%(message)s")
    ch.setFormatter(ch_formatter)

    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    logger = get_mylogger(os.getcwd())
    # 日志
    logger.debug('this is a logger debug message')
    logger.info('this is a logger info message')
    logger.warning('this is a logger warning message')
    logger.error('this is a logger error message')
    logger.critical('this is a logger critical message')
    




