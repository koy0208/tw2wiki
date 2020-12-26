# tw2wiki

## Usage

```bash
# レポジトリをcolone
$ git clone https://github.com/kojiro0208/tw2wiki.git

# pythonディレクトリに入る
$ cd tw2wiki/python

# ファイルを分かち書きにしたファイルを作る
# data/in_make_dataに、trainとtest用の分かち書きしたファイルが作られる。
$ python wakati.py

# 分かち書きしたファイルを、model学習用に加工する。
# data/in_modelにarray形式の学習データと、単語→id、id→単語変換の辞書が保存される。
$ python make_train_data.py

# 学習
# 10epochごとにlearned_modelsにモデルが保存される。
$ python learning.py

# 予測結果確認
# テストデータの初め10この予測結果が出力される。
$ python predict.py
```
## Reference
[Seq2Seqを利用した文章生成](https://qiita.com/gacky01/items/064c0071e5cff87e5c08)

[PyTorchでAttention Seq2Seqを実装してみた](https://qiita.com/m__k/items/646044788c5f94eadc8d)

