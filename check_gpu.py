
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('sleep_time', default=3600, type=int, help='train config file path')
    args = parser.parse_args()
    return args


# 检查GPU使用情况
def check_gpu_usage():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
        usage = [int(x) for x in result.decode('utf-8').strip().split('\n')]
        return usage
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while checking GPU usage: {e}")
        return []

# 主函数
def main():
    import time
    flag = False
    while(flag is False):
        usage = check_gpu_usage()
        flag = (usage and all(u == 0 for u in usage))
        if flag:
            args = parse_args()
            sleep_time = args.sleep_time
            time.sleep(sleep_time)
            # time.sleep(3600)       # 1h 等待1个小时看看是不是他们真的不用了
            usage = check_gpu_usage()
            flag = (usage and all(u == 0 for u in usage))
            print('-----------------------')
            print(' GPUs available')
        else:           
            print('None GPUs available')
            time.sleep(60)       # 1min   

if __name__ == "__main__":
    main()


