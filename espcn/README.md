# README

# 使い方

Train
```
python train.py --dataset_dir [データセットのパス] 
```

モデルの使用
```
python expand.py [対象となるディレクトリ、またはファイル] [出力するディレクトリ]
```
＊同じディレクトリを入力と出力にすると、画像が上書きされるので注意してください

データセットのディレクトリ構造は以下のようにしてください
dataset 
 ├train
 │ ├train用の画像1
 │ ├train用の画像２
 │   :
 │   :
 └test
   ├test用の画像1
   ├test用の画像２
   ├ :
