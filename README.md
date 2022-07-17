<!-- ![image](https://user-images.githubusercontent.com/42250266/179401186-210643bd-0549-4ac1-af9c-318de926e321.png)
 -->
<img src="https://user-images.githubusercontent.com/42250266/179401955-62c63122-1471-40cf-9ccb-61a266121392.png" width="100%" height="100%">

<!-- <img src="https://user-images.githubusercontent.com/42250266/179400316-d857dde1-e933-441e-8d0b-fa337bcbea67.png" width="100%" height="100%"> -->
<!-- ![image](https://user-images.githubusercontent.com/42250266/179400316-d857dde1-e933-441e-8d0b-fa337bcbea67.png)
 -->

<!---
![image](https://user-images.githubusercontent.com/42250266/179398735-9153be4e-ec85-42c0-9583-c7002e0dc545.png)

-->
# Mapping and Tracking Sentiment Arcs in Social Media Streams

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


### Results

We applied the model to the series of COVID-19 tweets and plotted the average daily sentiment scores against time: 
![MeanperdayCOVID](https://user-images.githubusercontent.com/42250266/179397899-796c9718-a65c-4a13-a7fb-e97e7bbd6050.png)

But the daily averages were highly noisy, so we used a smoothing filter to see the overall trend:
![MeanperdaySAVGOLCOVID](https://user-images.githubusercontent.com/42250266/179397929-5ee411a4-0556-47d7-b3fc-c024d3abe358.png)

In comparison to [daily death rates as reported by WHO](https://covid19.who.int/data):
![deathratesCOVID](https://user-images.githubusercontent.com/42250266/179399007-302f15ec-95d0-470e-8b47-b92d01436714.png)



If we add some key events to the plot to see how they line up with the trends, we will see that the average sentiment scores were negative at the start of the pandemic. The feelings of fear might explain the initial trend. 
Another slight drop in sentiments was observed during the summer of 2020 when the second pandemic wave started. But as the vaccination campaigns started, there was a slight increase in sentiment scores:

![image](https://user-images.githubusercontent.com/42250266/179398060-5a9160ec-eaee-487f-bed6-6dc7f440eafe.png)


**Vaccinations**

We wanted to investigate the effects of the vaccination campaigns in more detail, so we used a [dataset available on Kaggle](https://www.kaggle.com/datasets/gpreda/all-covid19-vaccines-tweets) that contained tweets about the different vaccines we also extracted tweets from the COVID-19 dataset.
We then applied the sentiment analysis model and plotted the trend along with the COVID-19 sentiment arc:

 - *Plot of vaccination tweets extracted from the COVID-19 dataset*:
	 ![inhouse](https://user-images.githubusercontent.com/42250266/179398335-b18ba281-6958-48c1-8e1b-3b87b053340f.png)

 - *Plot of vaccination tweets from the Kaggle dataset*:
    ![external](https://user-images.githubusercontent.com/42250266/179398340-75c42ec5-9801-46fa-b73b-142a2435c73c.png)

 - *Plot of both datasets appended*:
    ![all](https://user-images.githubusercontent.com/42250266/179398350-cfd75717-d25f-49c6-98df-4d0002dc7656.png)

Therefore, we could infer that the sentiment arc for COVID-19 was influenced by the introduction of vaccines, or other factors that have not been investigated yet.

**Contextual Meaning of Words**

Another point we were curious about was how context could change a word's polarity and how it would affect the arc. The word "positive" usually means something pleasant when describing a situation or experience and should imply a positive sentiment. In the context of COVID-19, a positive test result means that someone has been infected, which might indicate a negative sentiment.

<img src= "https://user-images.githubusercontent.com/42250266/179398475-0d1e9efe-22c1-41b9-b5d3-ee3929e280da.jpg" width="550">

A phrase like "I have been tested positive" should be classified as negative. But, since our model was pre-trained and fine-tuned on a general domain corpus, it classifies it as positive.

We tested an approach where we replaced the word "positive" with "infected" in all tweets in the corpus that did not contain the words "impact" or "effect". Similarly, we replaced "negative" with "free".

```
if (not 'impact' in text.lower()) or (not 'effect' in text.lower()):
  text = re.sub('negative|Negative',' free ', text)
  text = re.sub('positive|Positive',' infected ',text)
```

We plot the sentiment arc resulting from this approach in comparison to the original COVID-19 arc. The modified arc is slightly more negative however the difference is almost unnoticeable showing that the change in word meanings doesn’t affect the model drastically:
![MeanperdaySAVGOLCOMPARISONCOVID](https://user-images.githubusercontent.com/42250266/179398591-2a0194b3-91b7-49ee-af14-5d5bc79604b6.png)

**Happiness Scores**

We carried out further analysis using the [Hedonometer](http://hedonometer.org/). While it is a fact that the models measure different qualities of the text in different ways and on different scales, we were curious to see if the two arcs - one based on happiness scores and one based on sentiment scores - would exhibit roughly the same pattern.

We plot both arcs on the same plot, each with different scoring scales: the Hedonometer scale is 1 (very negative) to 9 (very positive), with 5 being neutral; our
scale is -1 (very negative), +1 (very positive) and 0 neutral. Both arcs convey
the overall daily average scores:

![hedon2](https://user-images.githubusercontent.com/42250266/179398930-8fa76e70-4214-4d1b-9b2a-d8a7c153d0ca.png)

This is an isolated arc of the Hedonometer to see the fluctuations in more detail:
![hedon1](https://user-images.githubusercontent.com/42250266/179398945-0b7d0107-de4b-43ce-b12b-10bcb9cb32ff.png)


## Topic Modelling

We explored topics discussed in tweets using [GSDMM](https://github.com/rwalk/gsdmm). We experimented with setting the
upper bound of the GSDMM with different numbers of topics. However, we finally chose
a model with 9 topics among all models because it showed diverse and less redundant
topics when manually examined. The alpha and beta parameters are set to 0.1 as used in
the original paper, and the number of iterations is set to 30.

We applied the GSDMM algorithm on tweets posted during 3 key stages of the pandemic:

 1. The start of the outbreak (February - April 2020)
 2. The second wave (July - September 2020)
 3. The start of the vaccination campaign (December 2020 - February
    2021)

We obtained 9 different clusters of the words with the high probability of belonging to each cluster. We used our own judgement to label each cluster by 
manually inspecting samples of tweets. Finally, after assigning a label to each topic, we identified 7 themes of the discussed topics:

![image](https://user-images.githubusercontent.com/42250266/179399289-902d13c3-d6ec-4725-a64f-9d5c9dd4accc.png)
The discussions of each theme varied in frequency and in focus throughout the pandemic.

**Wordcloud per each stage:**

 1. ![wwc2342020](https://user-images.githubusercontent.com/42250266/179399374-3536ea2d-3597-4b98-9209-afe07163b6a1.png)
 2. ![wwc789](https://user-images.githubusercontent.com/42250266/179399377-c01a9507-2154-4041-b15d-580fe42edf88.png)
 3. ![wwc120102](https://user-images.githubusercontent.com/42250266/179399380-564357f8-d9e7-4ed2-8d13-5f000908a885.png)


## Application of the model on another domain: Elon Musk's Twitter Acquision
To prove that the proposed model is generic, we applied it to tweets from another domain.
So we collected tweets over almost 16 days - around the time Elon Musk announced he's offering to buy twitter - and used the sentiment analysis model to plot a sentiment arc:

<img src= "https://user-images.githubusercontent.com/42250266/179399626-cae35d65-5333-4305-aa1b-e389a1174fa8.png" width="550">

![elonsent](https://user-images.githubusercontent.com/42250266/179399592-dafbb16a-dd6c-4b7a-81f6-49018503f416.png)


# Libraries Used:
 - numpy
 - pandas
 - vaex
 - sklearn
 - matplotlib
 - seaborn
 - codecs
 - re
 - os
 - transformers
 - torch
 - random
 - datetime
 - tqdm
 - tensorflow
 - nltk
 - langdetect
 - statsmodels
 - scipy
 - wordcloud
 - gsdmm
 - gensim
 - labMTsimple
 - marisa_trie
 
 
 
