import nltk
import re
import heapq
import numpy as np
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

try:
	nltk.data.find('tokenizers/punkt')
except LookupError:
	print('nltk models has not been downloaded!! -> process downloading models')
	nltk.download()

paragraph = """
			Thank you all so very much. Thank you to The Academy. I have to congratulate the other incredible nominees in this room for their unbelievable performances.
			The Revenant was the product of an unbelievable cast and crew I got to work alongside. First off, to my brother in this endeavor, Mr. Tom Hardy. Tom, your fierice talent on screen can only be surpassed off screen by Mr. Alejandro Inarritu. As the history of cinema unfolds, you have forged your way into history this past two years. What an unbelievable talent you are.
			Thank you to you and Emmanuel Chivo Lubezki for creating a transcendent cinematic experience for all of us.
			Thank you to everyone at Fox and Regency. Ana Melching, you were the champion of this endeavor.
			I have to thank everyone from the very onset of my career — Mr. Jones for casting me in my first film. Mr. Scorsese for teaching me so much about the cinematic art form. To Mr. Rick Yorn, thank you for helping me navigate this industry. And, to my parents, none of this would be possible without you, and, to my friends, I love you dearly, you know who you are.
			And lastly, I just want to say this, making The Revenant was about man's relationship to the natural world — the world that we collectively felt in 2015 as the hottest year in recorded history. Our production had to move to the southernmost tip of this planet just to be able to find snow. Climate change is real, it is happening right now, it is the most urgent threat facing our entire species, and we need to work collectively together and stop procrastinating.
			We need to support leaders around the world who do not speak for the big polluters or the big corporations, but who speak for all of humanity, for the indigenous peoples of the world, for the billions and billions of underprivileged people who will be most affected by this, for our children's children, and for those people out there whose voices have been drowned out by the politics of greed.
			I thank you all for th is amazing award tonight. Let us not take this planet for granted; I do not take this night for granted.
			"""

sentences = nltk.sent_tokenize(paragraph)

for i in range(len(sentences)):
	sentences[i] = sentences[i].lower()
	sentences[i] = re.sub(r'\W',' ', sentences[i]) # remove all non-word symbols like . , ; / ...
	sentences[i] = re.sub(r'\s+',' ', sentences[i]) # replace any spaces between words with one space
	sentences[i] = sentences[i].strip() # Trim all the spaces at the begin and the end of each sentence

# Creating the histogram
word2count = {}

for s in sentences:
	words = nltk.word_tokenize(s)
	for word in words:
		if word not in word2count.keys():
			word2count[word] = 1
		else:
			word2count[word] += 1

freq_words = heapq.nlargest(100,word2count,key=word2count.get)

# IDF Matrix
word_idfs = {}

for word in freq_words:
	doc_count = 0
	for s in sentences:
		if word in nltk.word_tokenize(s):
			doc_count += 1

	word_idfs[word] = np.log((len(sentences) / doc_count) + 1)

# TF Matrx
tf_matrix = {}

for word in freq_words:
	doc_tf = []
	for s in sentences:
		frequency = 0
		for w in nltk.word_tokenize(s):
			if w == word:
				frequency += 1
		
		tf_word = frequency / len(nltk.word_tokenize(paragraph))
		doc_tf.append(tf_word)
	tf_matrix[word] = doc_tf

# TF-IDF calculation
tfidf_matrix = []
for word in tf_matrix.keys():
	tfidf = []
	for value in tf_matrix[word]:
		score = value * word_idfs[word]
		tfidf.append(score)
	tfidf_matrix.append(tfidf)

# Finding name entities
# words = nltk.word_tokenize(paragraph)
# tagged_words = nltk.pos_tag(words)

# nameEntities = nltk.ne_chunk(tagged_words)
# nameEntities.draw()

# Finding tagged words
# words = nltk.word_tokenize(paragraph)
# tagged_words = nltk.pos_tag(words)
# word_tag = []
# for tw in tagged_words:
# 	word_tag.append(tw[0] + "_" + tw[1])

# Fiding stop words
# for i in range(len(sentences)):
# 	words = nltk.word_tokenize(sentences[i])
# 	new_words = [word for word in words if word not in stopwords.words('english')]
# 	sentences[i] = ' '.join(new_words)

# Lemmatize
# lemmatizer = WordNetLemmatizer()
# for i in range(len(sentences)):
# 	words = nltk.word_tokenize(sentences[i])
# 	new_words = [lemmatizer.lemmatize(word) for word in words]
# 	sentences[i] = ' '.join(new_words)

# Stem
# stemmer = PorterStemmer()
# for i in range(len(sentences)):
# 	words = nltk.word_tokenize(sentences[i])
# 	new_words = [stemmer.stem(word) for word in words]
# 	sentences[i] = ' '.join(new_words)