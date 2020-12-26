import pandas as pd
import csv
from sudachipy import tokenizer
from sudachipy import dictionary

tokenizer_obj = dictionary.Dictionary().create()
tokenizer_obj_tokenize = tokenizer_obj.tokenize
modeC = tokenizer.Tokenizer.SplitMode.C
df = pd.read_table('../data/in_make_data/sample_wiki_datamart.tsv')

def wakati(sent):
    words = tokenizer_obj_tokenize(sent, modeC)
    words_join = ' '.join([w.surface() for w in words])
    return words_join

def main():
    #テストデータ作成
    print('start wakati test input')
    test = pd.DataFrame()
    test['input'] = [wakati(sent) for sent in df['input'][-1000:]]
    print('start wakati test output')
    test['output'] = [wakati(sent) for sent in df['output'][-1000:]]
    print('write test')
    test.to_csv('../data/in_make_data/sample_test_data.tsv',sep='\t', index=False)
    del test

    #学習データ作成
    print('start wakati train input')
    train = pd.DataFrame()
    train['input'] = [wakati(sent) for sent in df['input'][:-1000]]

    print('start wakati train output')
    train['output'] = [wakati(sent) for sent in df['output'][:-1000]]
    print('write train')
    train.to_csv('../data/in_make_data/sample_train_data.tsv',sep='\t', index=False)
    del train

if __name__ == '__main__':
    print("Start if __name__ == '__main__'")
    print('call main()')
    main()

