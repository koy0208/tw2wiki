import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

train = pd.read_table('../data/in_make_data/sample_train_data.tsv')

#input、outputの最長学習サイズ、出現頻度最小回数を指定
input_len = 30
output_len = 100
#最小単語出現頻度を定義
min_freq = 2
#辞書を作成する。
word2id={}
id2word={}
count_dict = {}
#単語分割した後のデータ格納list
input_train_list = []
output_train_list = []

def main():
    for i in range(len(train['input'])):
        # 学習データ用に単語に分割してておく
        #最初の3センテンスを使用
        words_input = '。'.join(train['input'][i].split('。')[0:3])
        words_input = words_input.split(' ')
        words_output = '。'.join(train['output'][i].split('。')[0:3])
        words_output = ('_'+' '+words_output).split(' ')
        # 単語数が少なすぎるデータは除く
        if (len(words_input)<=10)|(len(words_output)<=20):
            continue
        # 最大単語数を指定する。
        if len(words_input)>=input_len:
            words_input = words_input[0:(input_len)]
        if (len(words_output)>=output_len):
            words_output = words_output[0:(output_len)]
        # inputを逆順にする。
        words_input.reverse()
        input_train_list.append(words_input)
        output_train_list.append(words_output)

        # 単語を一つずつ取り出す
        # 単語の数も数える
        for word in words_input:
            #辞書に単語がなければ加える
            if word not in word2id:
                idx = len(word2id)
                word2id[word] = idx
                id2word[idx] = word
                count_dict[word] =1
            else:
                count_dict[word] +=1
        #outputも同様
        for word in words_output:
            #辞書に単語がなければ加える
            if word not in word2id:
                idx = len(word2id)
                word2id[word] = idx
                id2word[idx] = word
                count_dict[word] =1
            else:
                count_dict[word] +=1
    #もう一度辞書を作り直す。
    word2id_new = {}
    id2word_new = {}
    # padding用の文字と頻度が少ない文字の置き換えを辞書のはじめに加える。
    word2id_new['<pad>'] = 0
    id2word_new[0] = '<pad>'
    word2id_new['<UNK>'] = 1
    id2word_new[1] = '<UNK>'
    #出現頻度が一定数以上あれば、辞書に加える。
    for word in word2id.keys():
        if count_dict[word] >= min_freq:
            idx = len(word2id_new)
            word2id_new[word] = idx
            id2word_new[idx] = word

    #学習データ数
    print('inputdata {}'.format(len(input_train_list)), 
            'outputdata {}'.format(len(output_train_list)))
    #語彙数
    print('vocab len {}'.format(len(word2id_new)))
    #文章の長さが短い場合は<pad>で埋める。
    #入れ物用の箱を作る
    input_words_pad = np.array([['<pad>']*input_len]*len(input_train_list), dtype=object)
    for i in tqdm(range(len(input_train_list))):
        for j in range(len(input_train_list[i])):
            word = input_train_list[i][j]
            #頻度少ない単語はUNKとする。
            if word not in word2id_new:
                word = '<UNK>'
            input_words_pad[i][j] = word

    #outputも同様な処理
    output_words_pad = np.array([['<pad>']*output_len]*len(output_train_list), dtype=object)
    for i in tqdm(range(len(output_train_list))):
        for j in range(len(output_train_list[i])):
            word = output_train_list[i][j]
            #頻度少ない単語はUNKとする。
            if word not in word2id_new:
                word = '<UNK>'
            output_words_pad[i][j] = word

    #outputにUNKを含む場合は、データを除く
    no_unk_index = [i for i in tqdm(range(len(output_words_pad))) if '<UNK>' not in output_words_pad[i]]
    input_words_pad = input_words_pad[no_unk_index]
    output_words_pad = output_words_pad[no_unk_index]
    print('last inputdata {}'.format(len(input_words_pad)), 
            'last outputdata {}'.format(len(input_words_pad)))

    #単語をid化してsaveする。
    input_data = np.empty_like(input_words_pad)
    output_data = np.empty_like(output_words_pad)

    for i in tqdm(range(len(input_words_pad))):
        input_data[i,:] = np.array([word2id_new[c] for c in input_words_pad[i]])
        output_data[i,:] = np.array([word2id_new[c] for c in output_words_pad[i]])


    print('save data')
    np.save('../data/in_model/sample_input_data.npy', input_data.astype('int'))
    np.save('../data/in_model/sample_output_data.npy', output_data.astype('int'))

    print('save dict')
    with open('../data//in_model/sample_word2id.pkl', 'wb') as f:
        pickle.dump(word2id_new, f)
    with open('../data//in_model/sample_id2word.pkl', 'wb') as f:
        pickle.dump(id2word_new, f)

if __name__ == "__main__":
    print('make train data')
    main()
