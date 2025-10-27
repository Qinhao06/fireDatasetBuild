#!/usr/bin/env python3
import subprocess
import sys
import os

def start_labelimg():
    try:
        # 尝试启动labelImg
        subprocess.run([sys.executable, '-m', 'labelImg'], check=True)
    except subprocess.CalledProcessError:
        print("labelImg启动失败，尝试直接运行...")
        try:
            subprocess.run(['labelImg'], check=True)
        except:
            print("无法启动labelImg，请检查安装是否成功")
            print("手动安装命令: pip install labelImg")

if __name__ == "__main__":
    print("🏷️  启动 LabelImg 标注工具...")
    start_labelimg()
