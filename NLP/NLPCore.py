import nltk

from nltk import PorterStemmer, WordNetLemmatizer

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

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

for i in range(len(sentences)):
	words = nltk.word_tokenize(sentences[i])
	new_words = [lemmatizer.lemmatize(word) for word in words]
	sentences[i] = ' '.join(new_words)

print(sentences)

for i in range(len(sentences)):
	words = nltk.word_tokenize(sentences[i])
	new_words = [stemmer.stem(word) for word in words]
	sentences[i] = ' '.join(new_words)