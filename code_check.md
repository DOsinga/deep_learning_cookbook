- [x] 03.1 Using pre trained word embeddings.ipynb
   - 8G RAM では almost_similar の計算で Memory Error。 15GであればOK
- [x] 03.2 Domain specific ranking using word2vec cosine distance.ipynb
   - Have to change the position of figsize
   - 15GではMemoryError。メモリが十分であればまわる(はず。たしかColaboratoryではまわった)
- [ ] 04.1 Collect movie data from Wikipedia.ipynb
   - import re が必要
   - Takes very long time and sometimes make OS un accessible.
     - GCE w/ 15RAM では、12時間ほどまわすと接続不可となった...
- [x] 04.2 Build a recommender system based on outgoing Wikipedia links.ipynb
- [] 05.1 Generating Text in the Style of an Example Text.ipynb
   - Have to add os.makedirs('zoo/06')
   - zoo/06? zoo/05 ではないか?（もともと6章想定だった時のなごりでは?）
   - gutenbergのあるなしでコードを分けているが、これ必要??
   - generate_code で意味のあるコードが生成されない
   - st という変数が定義されずに使われている
   - 学習に5時間ほどかかる
   - 以下のwarningがでる
   ```
   `imresize` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
    Use ``skimage.transform.resize`` instead.
    ```
- [x] 06.1 Question matching.ipynb
- [ ] 07.1 Text Classification.ipynb
   - NameError: name 'inverse_vocab' is not defined。以下を追加
     ```
     inverse_vocab = {v: k for k, v in tfidf_vec.vocabraly_.items()}
     ```
    - np is not defined
    - random_forrest > random_forest
    - max_sequence_len が定義されていない > そもそも次のcellにコピペされているようなので、cellごと削除
    - name 'regularizers' is not defined > from keras import regularizers を追加
    - name 'training_count' is not defined > 追加
    - twitter / emoji がない
- [ ] 07.2 Emoji Suggestions.ipynb
   - twitter / emoji がない。 pip install twitter / emoji
   - 本文にあわせ data/emojis.csv を data/emojis.txt に変更
   - `[x['text'] for x in itertools.islice(status_stream.sample(), 0, 5) if x.get('text')]` だと、ほとんど空のリストになる。if x.get('text') してからisliceでは?
   - `len(train_tweets) / BATCH_SIZE` は Python3 では浮動小数となるので `len(train_tweets) // BATCH_SIZE` に変更
   - `os.makedirs('zoo/07', exist_ok=True)` を追加
   - `"75s - loss: 2.3855 - acc: 0.4368\n[2.8089022636413574, 0.38840296648550726]"` なにこれ？
   - 'data/twitter_w2v.model' が存在しない > 学習コードを追加
   - 'w2v_model.syn0' > 'w2v_model.wv.syn0'
   - 'model[...]' > 'model.wv[...]'
   - 'in model:' > 'in model.wv:'
   - 日本語の取扱に関する記述があっても良いかも
- [ ] 07.3 Tweet Embeddings.ipynb
- [ ] 08.1 Sequence to sequence mapping.ipynb
    - ライブラリのインストールが必要
      - gutenberg / nltk / inflect
    - ValueError: Input 0 is incompatible with layer repeat_vector_2: expected ndim=2, found ndim=3
      - そもそも repeat = RepeartVector の repeat 使っていない。
      - num_chars も存在しないし、 len(chars) の chars はグローバル変数
    - seq2seq の部分はゼロから作り直す必要がありそう。というか作者ちゃんと理解していないのでは?
- [ ] 08.2 Import Gutenberg.ipynb
      - gutenberg が必要
- [ ] 08.3 Subword tokenizing.ipynb
      - import error??
- [x] 09.1 Reusing a pretrained image recognition network.ipynb
- [ ] 09.2 Images as embeddings.ipynb
    - Flickr キーが必要
- [ ] 09.3 Retraining.ipynb
- [ ] 10.1 Building an inverse image search service.ipynb
    - JSON Decode Error
- [ ] 11.1 Detecting Multiple Images.ipynb
    - keras_frcnn が必要だが...
- [ ] 12.1 Activation Optimization.ipynb
    - data/pilatus800.jpg がない
- [ ] 12.2 Neural Style.ipynb
    - ResourceExhaustedError (GPU のOOM)
- [ ] 13.1 Quick Draw Cat Autoencoder.ipynb
- [x] 13.2 Variational Autoencoder.ipynb
- [ ] 13.5 Quick Draw Autoencoder.ipynb
- [ ] 14.1 Importing icons.ipynb
     - svglib
- [ ] 14.2 Icon Autoencoding.ipynb
     - icons/index.json がない
- [ ] 14.2 Variational Autoencoder Icons.ipynb
     - icons/index.json がない
- [ ] 14.3 Icon GAN.ipynb
     - cv2がない
- [ ] 14.4 Icon RNN.ipynb
     - icons/index.json がない
- [ ] 15.1 Song Classification.ipynb
     - /home/douwe/genres がない
- [ ] 15.2 Index Local MP3s.ipynb
     - tinytag がない
- [ ] 15.3 Spotify Playlists.ipynb
- [ ] 15.4 Train a music recommender.ipynb
- [ ] 16.1 Productionize Embeddings.ipynb
     - psycopg2 がない
- [ ] 16.2 Prepare Keras model for Tensorflow Serving.ipynb
    - tensorflow_serving がない
- [ ] 16.3 Prepare model for iOS.ipynb
    - coremltools がない
- [x] 16.4 Simple Text Generation.ipynb
    - keras_js ディレクトリが存在しない
    ```
    import os
    os.makedirs('keras_js', exist_ok=True)
    ```
- [ ] Simple Seq2Seq.ipynb
