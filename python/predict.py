import pickle
import numpy as np
import pandas as pd
import torch
from sudachipy import tokenizer
from sudachipy import dictionary
from models import Encoder, AttentionDecoder
from functions import get_max_index

# データ読み込み
test = pd.read_table('../data/in_make_data/sample_test_data.tsv')
#辞書読み込み
with open('../data/in_model/sample_id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)

with open('../data/in_model/sample_word2id.pkl', 'rb') as f:
    word2id = pickle.load(f)

#変数定義
input_len = 50
output_len = 100
embedding_dim = 200
hidden_dim = 192
BATCH_NUM=1
vocab_size = len(word2id)
device = 'cpu'


# エンコーダーの設定
encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
attn_decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, BATCH_NUM).to(device)

#学習済みモデルの読み込み
en_model_path = '../learned_models/sample_en_model_e30'
de_model_path = '../learned_models/sample_de_model_e30'

encoder.load_state_dict(torch.load(en_model_path))
attn_decoder.load_state_dict(torch.load(de_model_path))

# 分かち書きトークナイザーの設定
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

def predict(text):
  #inputdataの作成
  input_data = np.zeros([1,input_len], dtype=int)
  #単語分割されたリストにする。
  wakati_text =  [m.surface() for m in tokenizer_obj.tokenize(text, mode)]
  if len(wakati_text)>=50:
        wakati_text = wakati_text[0:50]
  #入力を逆順にする。
  wakati_text.reverse()
  #知らない単語は<UNK>に置き換える
  for j in range(len(wakati_text)):
      word = wakati_text[j]
      if word in word2id:
          input_data[0][j] = word2id[word]
      else:
          input_data[0][j]  = word2id['<UNK>']

  #予測
  with torch.no_grad():
    input_tensor = torch.tensor(input_data, device=device)
    hs, encoder_state = encoder(input_tensor)

    # Decoderにはまず文字列生成開始を表す"_"をインプットにするので、"_"のtensorをバッチサイズ分作成
    start_char_batch = [[word2id["_"]] for _ in range(1)]
    decoder_input_tensor = torch.tensor(start_char_batch, device=device)

    decoder_hidden = encoder_state
    batch_tmp = torch.zeros(BATCH_NUM,1, dtype=torch.long, device=device)
    # 予測文字を取得しつつ、そのまま次のdecoderのインプットとなる
    for _ in range(output_len - 1):
        decoder_output, decoder_hidden, _ = attn_decoder(decoder_input_tensor, hs, decoder_hidden)
        # 予測文字を取得しつつ、そのまま次のdecoderのインプットとなる
        decoder_input_tensor = get_max_index(decoder_output.squeeze().reshape(1,-1), BATCH_NUM)
        #decoder_input_tensor = get_max_index(decoder_output.squeeze())
        batch_tmp = torch.cat([batch_tmp, decoder_input_tensor], dim=1)

  return ''.join([id2word[i.item()] for i in batch_tmp[:,1:][0]]).replace('<pad>','')

def main():
    for sent in test['input'][0:10]:
        out =  predict(sent)
        print('')
        print('In : ', sent.replace(' ',''))
        print('Out : ', out)

if __name__ == "__main__":
  main()