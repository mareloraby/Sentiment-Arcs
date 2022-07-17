

<img src="https://user-images.githubusercontent.com/42250266/179395826-8aa09a83-2da6-47c5-8ad5-18ae38416245.png" width="100%" height="100%">

# Mapping and Tracking Sentiment Arcs on Social Media Streams

This project aims to track people's opinions towards a major event, and plot a **sentiment arc**. This could be achieved through **sentiment analysis** which is the process of identifying opinions expressed in a text to determine whether a writer's attitude towards a particular topic is positive, negative, or neutral, and then using time series analysis to plot the sentiments against time and identify patterns and trends.

We create a sentiment analysis model to classify social media posts about a specific topic and plot a sentiment arc. We aim to make it **generic** in the sense that it can be applied to any domain rather than just limiting it to the one domain.

We also try to explain the shape of the arcs by extracting other discussed topics and comparing the temporal variations to actual events that could have influenced the shape.


## Solution Approach
![image](https://user-images.githubusercontent.com/42250266/179396873-917e306f-29e4-477f-ab20-d2f06f217a86.png)

We chose to work with Twitter, and our method begins with fetching a series of tweets related to a specific topic, then pre-processing the data by removing noise and redundant information. Later, the cleaned data is passed through a sentiment analysis model. The results are then plotted against time to create a sentiment arc. And after another phase of pre-processing, a text clustering algorithm (GSDMM) is applied to identify discussed topics from the data. 

![image](https://user-images.githubusercontent.com/42250266/179397267-f44ff00e-5967-4435-a734-a91d1b9534f0.png)

For the sentiment analysis model, we fine-tune BERT language model on the [TweetEval benchmark for sentiment analysis](https://github.com/cardiffnlp/tweeteval).
BERT outputs vectors of size hidden_size for each input token
in a sequence, starting with **[CLS]** and separated by **[SEP]**. BERT takes the final hidden state h of the [CLS] token as the representation of the whole sequence.

We used the [HuggingFace 🤗 Transformers](https://huggingface.co/transformers) library, which provides a PyTorch interface to
fine-tune pre-trained language models. We specifically used BERT-Base-cased pre-trained
model.
Before passing the input sequences to BERT encoder, we needed to specify the maximum
length of our sentences for padding/truncating to. So we performed one tokenization pass of the dataset to store the lengths of each
tokenized tweet, then we plotted the distributions only to find that most of the tweets
contained less than 70 tokens. To be on the safe side, we set the maximum length to 85.

According to the authors of [BERT paper](https://arxiv.org/abs/1810.04805), fine-tuning for classification tasks can be achieved by adding only one output layer, so a minimal number of parameters need to be learned. Thus, we add a simple
single-hidden-layer feed-forward neural network, with softmax to the top of BERT to
estimate the probability of a label *c* as:


$$
    p(c|\textbf{h}) = \textrm{softmax}(W\textbf{h})
$$

where W denotes the task-specific parameter matrix, and h denotes the final hidden
state of **[CLS]** token.

![image](https://user-images.githubusercontent.com/42250266/179397308-dbc78f6b-b1cd-48a6-ae1b-b2bb2032eecb.png)


We used these probabilities as sentiment intensities, and we scaled them to cover a range from -1 to +1. Tweets that were estimated to belong to the negative class had their confidence scores multiplied by -1 to cover a range from -1 to 0. while tweets that were estimated to belong to the positive have their scores left as they are to cover the range from 0 to +1. Neutral sentiments' scores are all mapped to 0.

## Applying the model on COVID-19 tweets.

[Covid-19 Twitter chatter dataset for scientific use](https://github.com/thepanacealab/covid19_twitter) is an ongoing project dating
back to January 2020, where researchers in Georgia State University’s Panacea Lab are
collecting COVID-19 tweets from the publicly available Twitter stream. 
As per Twitter’s Terms of Service, which do not allow the full JSON for datasets of
tweets to be distributed to third parties, the dataset contained only tweet identifiers(IDs).
In this project, and due to the limited computation power at hand, we worked with a subset of 100,000 tweets per month. The dataset was too big to deal with, so we split it by months and extracted only English tweets by applying a filter on the “lang” column provided. Then we saved only the tweets’ IDs. Handling a dataset of this size was made feasible through the use of [Vaex](https://vaex.io/docs/index.html).

To get the raw tweet text from the tweet IDs, we selected a random sample of 100,000 tweets from 16 months starting from February 2020 till June 2021, and [hydrated](https://stackoverflow.com/questions/34191022/what-does-hydrate-mean-on-twitter) them using [Hydrator app](https://github.com/DocNow/hydrator). The dataset obtained from Hydrator contained 35 features, but we worked on only two: “full text”, which is the full, raw text of the tweet; and “Datetime”, which is the date and time when the tweet was posted.


## Results


