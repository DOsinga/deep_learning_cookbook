# Deep Learning Cookbook Notebooks

This repository contains 35 python notebooks demonstrating most of the key
machine learning techniques in Keras. The notebooks accompany the book
[Deep Learning Cookbook](https://www.amazon.com/Deep-Learning-Cookbook-Practical-Recipes) but work well on their own. A GPU is not required to run them,
but on a mere CPU things will take quite a while.

## Getting started

To get started, setup a virtual env, install the requirements and start the notebook server:

```Bash
git clone https://github.com/DOsinga/deep_learning_cookbook.git
cd deep_learning_cookbook
python3 -m venv venv3
source venv3/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## The notebooks

#### [03.1 Using pre trained word embeddings](https://github.com/DOsinga/deep_learning_cookbook/blob/master/03.1%20Using%20pre%20trained%20word%20embeddings.ipynb)

In this notebook, we'll use a pretrained word embedding model (Word2Vec) to explore how word embeddings allow us
to explore similarities between words and relationships between words. For example, find the capital of a country
or the main products of a company. We'll finish with a demonstration of using t-SNE to plot high dimensional
spaces on a 2D graph. 

#### [03.2 Domain specific ranking using word2vec cosine distance](https://github.com/DOsinga/deep_learning_cookbook/blob/master/03.2%20Domain%20specific%20ranking%20using%20word2vec%20cosine%20distance.ipynb)

Building on the previous recipe, we'll use the distances between the words to do domain specific rankings. Specifically
we'll look at countries. First we create a small classifier to find all countries in the set of words, based on a small
sample. We'll then use a similar approach to show relevance for specific words for countries. For example, since
cricket is closer to India than to Germany, cricket is probably more relevant. We can plot this on a world map which
lights up countries based on their relevance for specific words.

#### [04.1 Collect movie data from Wikipedia](https://github.com/DOsinga/deep_learning_cookbook/blob/master/04.1%20Collect%20movie%20data%20from%20Wikipedia.ipynb)

This notebook shows how to download a dump of the Wikipedia and parse it to extract structured data by using the
category and template information. We'll use this to create a set of movies including rating data.

#### [04.2 Build a recommender system based on outgoing Wikipedia links](https://github.com/DOsinga/deep_learning_cookbook/blob/master/04.2%20Build%20a%20recommender%20system%20based%20on%20outgoing%20Wikipedia%20links.ipynb)

Based on the structured data extracted in the previous notebook, we'll train a network that learns to predict a movie
based on the outgoing links on the corresponding Wikipedia page. This creates embeddings for the movies. This in
turn lets us recommend movies based on other movies - similar movies are next to each other in the embedding
space.

#### [05.1 Generating Text in the Style of an Example Text](https://github.com/DOsinga/deep_learning_cookbook/blob/master/05.1%20Generating%20Text%20in%20the%20Style%20of%20an%20Example%20Text.ipynb)

We train an LSTM to write Shakespeare. We'll follow this up with one that generates Python code by training a similar
LSTM on the Python system codebase. Visualizing what the network has learned shows us what the Python producing
network is paying attention to as it produces or read Python code.

#### [06.1 Question matching](https://github.com/DOsinga/deep_learning_cookbook/blob/master/06.1%20Question%20matching.ipynb)

In this notebook we train a network to learn how to match questions and answers from stackoverflow; this sort of indexing
than allows us to find given a question what the most likely answer in a database is. We try a variety of approaches to
improve upon the first not terribly great results.

#### [07.1 Text Classification](https://github.com/DOsinga/deep_learning_cookbook/blob/master/07.1%20Text%20Classification.ipynb)

This notebook shows eight different machine learning approaches to classify texts into a variety of sentiments. The first
three are classical learners, followed by a number of deep learning models, character or word based and lstm vs cnn. The
best approach is to combine all approaches in one model.

#### [07.2 Emoji Suggestions](https://github.com/DOsinga/deep_learning_cookbook/blob/master/07.2%20Emoji%20Suggestions.ipynb)

We start by harvesting a large set of tweets and we keep the ones that contain exactly one emoji (you can skip this step,
a training set is included). We then train a number of deep models to use the tweet minus the emoji to predict the missing
emoji. We end up effectively with a model that can find the best emoji for a given bit of text.

#### [07.3 Tweet Embeddings](https://github.com/DOsinga/deep_learning_cookbook/blob/master/07.3%20Tweet%20Embeddings.ipynb)

Some experimental code (not included in the book) to semantically index tweets such that tweets that are similar show
up next to each other; effectively doing what Word2Vec does for words, but now for tweets.

#### [08.1 Sequence to sequence mapping](https://github.com/DOsinga/deep_learning_cookbook/blob/master/08.1%20Sequence%20to%20sequence%20mapping.ipynb)
#### [08.2 Import Gutenberg](https://github.com/DOsinga/deep_learning_cookbook/blob/master/08.2%20Import%20Gutenberg.ipynb)

Small notebook demonstrating how to download books from the Gutenberg project. Tokenizes a set of book in preparation of
the subword tokenizing in the next notebook.

#### [09.1 Reusing a pretrained image recognition network](https://github.com/DOsinga/deep_learning_cookbook/blob/master/09.1%20Reusing%20a%20pretrained%20image%20recognition%20network.ipynb)

Quick notebook demonstrating how to load a pretrained network and apply it on an image of, well, what else? a cat.
Shows how to normalize the image and decode the predictions.

#### [09.2 Images as embeddings](https://github.com/DOsinga/deep_learning_cookbook/blob/master/09.2%20Images%20as%20embeddings.ipynb)

In this notebook we use the Flickr API to fetch a feed of search results for the search term cat. By running each result through a pre-trained network
we get vectors that project the images in a 'space'. The center of that space in some way represents the most cat image possible. By reranking on
distance to that center we can weed out images that are less cat like. Effectively we can improve upon the Flickr search results without
knowing the content!

#### [09.3 Retraining](https://github.com/DOsinga/deep_learning_cookbook/blob/master/09.3%20Retraining.ipynb)
#### [10.1 Building an inverse image search service](https://github.com/DOsinga/deep_learning_cookbook/blob/master/10.1%20Building%20an%20inverse%20image%20search%20service.ipynb)
#### [11.1 Detecting Multiple Images](https://github.com/DOsinga/deep_learning_cookbook/blob/master/11.1%20Detecting%20Multiple%20Images.ipynb)

Use the fact that imag classification networks extract features per larger square sub-image to detect multiple dogs and cats in the same
image or at least to know where in the image you can find your cat or dog. The approach her is a lot simpler than what is the state of the
art, but also a lot easier to follow, so a good way to get started.

#### [12.1 Activation Optimization](https://github.com/DOsinga/deep_learning_cookbook/blob/master/12.1%20Activation%20Optimization.ipynb)
#### [12.2 Neural Style](https://github.com/DOsinga/deep_learning_cookbook/blob/master/12.2%20Neural%20Style.ipynb)
#### [13.1 Quick Draw Cat Autoencoder](https://github.com/DOsinga/deep_learning_cookbook/blob/master/13.1%20Quick%20Draw%20Cat%20Autoencoder.ipynb)
#### [13.2 Variational Autoencoder](https://github.com/DOsinga/deep_learning_cookbook/blob/master/13.2%20Variational%20Autoencoder.ipynb)
#### [13.5 Quick Draw Autoencoder](https://github.com/DOsinga/deep_learning_cookbook/blob/master/13.5%20Quick%20Draw%20Autoencoder.ipynb)
#### [14.1 Importing icons](https://github.com/DOsinga/deep_learning_cookbook/blob/master/14.1%20Importing%20icons.ipynb)
#### [14.2 Icon Autoencoding](https://github.com/DOsinga/deep_learning_cookbook/blob/master/14.2%20Icon%20Autoencoding.ipynb)
#### [14.2 Variational Autoencoder Icons](https://github.com/DOsinga/deep_learning_cookbook/blob/master/14.2%20Variational%20Autoencoder%20Icons.ipynb)
#### [14.3 Icon GAN](https://github.com/DOsinga/deep_learning_cookbook/blob/master/14.3%20Icon%20GAN.ipynb)
#### [14.4 Icon RNN](https://github.com/DOsinga/deep_learning_cookbook/blob/master/14.4%20Icon%20RNN.ipynb)
#### [15.1 Song Classification](https://github.com/DOsinga/deep_learning_cookbook/blob/master/15.1%20Song%20Classification.ipynb)
#### [15.2 Index Local MP3s](https://github.com/DOsinga/deep_learning_cookbook/blob/master/15.2%20Index%20Local%20MP3s.ipynb)
#### [15.3 Spotify Playlists](https://github.com/DOsinga/deep_learning_cookbook/blob/master/15.3%20Spotify%20Playlists.ipynb)
#### [15.4 Train a music recommender](https://github.com/DOsinga/deep_learning_cookbook/blob/master/15.4%20Train%20a%20music%20recommender.ipynb)
#### [16.1 Productionize Embeddings](https://github.com/DOsinga/deep_learning_cookbook/blob/master/16.1%20Productionize%20Embeddings.ipynb)
#### [16.2 Prepare Keras model for Tensorflow Serving](https://github.com/DOsinga/deep_learning_cookbook/blob/master/16.2%20Prepare%20Keras%20model%20for%20Tensorflow%20Serving.ipynb)
#### [16.3 Prepare model for iOS](https://github.com/DOsinga/deep_learning_cookbook/blob/master/16.3%20Prepare%20model%20for%20iOS.ipynb)
#### [16.4 Simple Text Generation](https://github.com/DOsinga/deep_learning_cookbook/blob/master/16.4%20Simple%20Text%20Generation.ipynb)
#### [Simple Seq2Seq](https://github.com/DOsinga/deep_learning_cookbook/blob/master/Simple%20Seq2Seq.ipynb)

Simple sequence-to-sequence mapping demo. The notebook shows how to teach a network how to form plurals.
