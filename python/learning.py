import numpy as np
import pandas as pd
import pickle 
import torch.nn as nn
import torch.optim as optim
import torch
from models import Encoder, AttentionDecoder
from functions import train2batch



with open('../data/in_model/sample_word2id.pkl', 'rb') as f:
    word2id = pickle.load(f)
with open('../data/in_model/sample_id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)

input_data = np.load('../data/in_model/sample_input_data.npy', allow_pickle=True)
output_data = np.load('../data/in_model/sample_output_data.npy', allow_pickle=True)


embedding_dim = 200
hidden_dim = 192
BATCH_NUM=10
EPOCH_NUM = 30
vocab_size = len(id2word)
device = 'cpu'

encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
attn_decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, BATCH_NUM).to(device)
# 損失関数
criterion = nn.CrossEntropyLoss()
# 最適化
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
attn_decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=0.001)

all_losses = []

def main():
    for epoch in range(1, EPOCH_NUM+1):
        epoch_loss = 0

        input_batch, output_batch = train2batch(input_data, output_data, batch_size=BATCH_NUM)
        for i in range(len(input_batch)):
            
            encoder_optimizer.zero_grad()
            attn_decoder_optimizer.zero_grad()

            # データをテンソルに変換
            input_tensor = torch.tensor(input_batch[i], device=device)
            output_tensor = torch.tensor(output_batch[i], device=device)
            
            # Encoder
            hs, h = encoder(input_tensor)
            # Decoder
            source = output_tensor[:, :-1]

            # Attention Decoderの正解データ
            target = output_tensor[:, 1:]

            loss = 0
            decoder_output, _, attention_weight= attn_decoder(source, hs, h)
            for j in range(decoder_output.size()[1]):
                loss += criterion(decoder_output[:, j, :], target[:, j])

            epoch_loss += loss.item()
            
            if (i%100) ==0:
                print("batch size {}: {}, loss: {}".format(len(input_batch), i, loss.item()))
            # 誤差逆伝播
            loss.backward()

            # パラメータ更新
            encoder_optimizer.step()
            attn_decoder_optimizer.step()

        # 損失を表示
        #10の倍数のエポックの時にモデルを保存
        if (epoch % 10) ==0:
            print('model save')
            en_model_path = '../learned_models/sample_en_model_e{}'.format(epoch)
            de_model_path = '../learned_models/sample_de_model_e{}'.format(epoch)
            torch.save(encoder.state_dict(), en_model_path)
            torch.save(attn_decoder.state_dict(), de_model_path)
        print("Epoch %d: %.2f" % (epoch, epoch_loss))
        all_losses.append(epoch_loss)
        if epoch_loss < 50: break
    print("Done")

if __name__ == "__main__":
    print("training ...")
    main()