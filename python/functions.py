import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle
import torch

# データをバッチ化するための関数を定義
def train2batch(input_data, output_data, batch_size=10):
    input_batch = []
    output_batch = []
    input_shuffle, output_shuffle = shuffle(input_data, output_data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i in range(0, len(input_data), batch_size):
      input_batch.append(input_shuffle[i:i+batch_size])
      output_batch.append(output_shuffle[i:i+batch_size])
    return input_batch, output_batch
 
 # Decoderのアウトプットのテンソルから要素が最大のインデックスを返す。つまり生成文字を意味する
def get_max_index(decoder_output, BATCH_NUM):
  results = []
  device ='cuda' if torch.cuda.is_available() else 'cpu'
  for h in decoder_output:
    results.append(torch.argmax(h))
  return torch.tensor(results, device=device).view(BATCH_NUM, 1)