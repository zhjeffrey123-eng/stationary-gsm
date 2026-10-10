"""双周期 PEC 像素超表面的谱域 EFIE 与驻值 GSM 提取。"""
import os
# 在 NumPy 首次导入前限制默认 BLAS 线程，用户已有设置优先。
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
__version__ = '2.0.0'
