import argparse
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from models.FourierGNN import FGN
import time
import os
import numpy as np

from exp.exp_main import Exp_Main
from utils.FourierGNN_utils import save_model, load_model, evaluate

parser = argparse.ArgumentParser(description='fourier graph network for multivariate time series forecasting')
parser.add_argument('--data', type=str, default='ECG', help='data set')
parser.add_argument('--feature_size', type=int, default='140', help='feature size')
parser.add_argument('--seq_length', type=int, default=12, help='inout length')
parser.add_argument('--pre_length', type=int, default=12, help='predict length')
parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--device', type=str, default='cuda:0', help='device')
