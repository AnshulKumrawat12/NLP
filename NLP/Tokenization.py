import nltk

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

# Get list in form of sentences
sentences = nltk.sent_tokenize(paragraph)

#Get list of all words including repeatation of words 
words = nltk.word_tokenize(paragraph)