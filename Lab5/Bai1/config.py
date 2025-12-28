import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "D:\\Python\\Nam3\\DL\\ThucHanh\\Lab5\\ViOCD\\train.json")
DEV_PATH   = os.path.join(BASE_DIR, "D:\\Python\\Nam3\\DL\\ThucHanh\\Lab5\\ViOCD\\dev.json")
TEST_PATH  = os.path.join(BASE_DIR, "D:\\Python\\Nam3\\DL\\ThucHanh\\Lab5\\ViOCD\\test.json")
SAVE_PATH  = os.path.join(BASE_DIR, "model_checkpoint.pth")

MIN_FREQ = 1  

# --- CẤU HÌNH TRAIN ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")