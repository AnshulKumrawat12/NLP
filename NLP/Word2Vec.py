# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:06:22 2021

@author: Anshul
"""

from gensim.models import Word2Vec
import re # Regular Expression
import nltk
from nltk.corpus import stopwords

paragraph= """ Let me now share with you, the enriching experience I had, while meeting more than
6000 farmers from different States and Union Territories visiting Rashtrapati Bhavan.
They evinced keen interest in the Mughal Gardens, the Herbal Gardens, the Spiritual
Garden, the Musical Garden, the Bio-diesel garden and the Nutrition Garden and interact
with the Horticultural specialists. Recently, during my address to the agricultural
scientists while participating in a National Symposium on “Agriculture Cannot Wait”, I
summarized the many practical suggestions given by farmers. We have to double the
agricultural production with reduced land, reduced water resources and reduced
manpower and improve the economic conditions of the nation through the principle of
“Seed to Food” since agriculture is the backbone of the nation. We should empower the
farmers to protect and nurture the fertile land for second green revolution. Meeting the
Scientists and the Farmers has given me the confidence
that the nation is poised to increase the agricultural GDP growth by atleast 4% per annum
through the partnership of farmers and agricultural scientists and industries particularly
for value addition. """


#Preprocessing and cleaning white spaces, special words, lowering
text = re.sub(r'\[[0-9]*\]', ' ',paragraph)
text = re.sub(r'\s+', ' ', text)
text = text.lower()
text = re.sub(r'\d', ' ', text)
text = re.sub(r'\s+', ' ',text)

sentences = nltk.sent_tokenize(text)

#tokenize sentences into words
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in set(stopwords.words('english'))]

#word2vec model
model = Word2Vec(sentences, min_count=1)

words = list(model.wv.index_to_key)

#Finding word vectors
vector = model.wv['economic']

#most similar words
similar = model.wv.most_similar('farmers')

