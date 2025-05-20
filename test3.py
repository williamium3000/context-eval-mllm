import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK data (only needs to be done once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Example sentence
sentence = "the image depicts a busy city street with a man standing on the sidewalk, talking on his cell phone. he is wearing a red jacket and appears to be engaged in a conversation. there are several other people walking along the sidewalk, some of them carrying handbags.\n\nthe street is filled with various vehicles, including cars, a truck, and a bicycle. some cars are parked along the side of the road, while others are driving or stopped in traffic. the presence of multiple people and vehicles creates a lively urban atmosphere."

# Predefined list of Multi-word Expressions (MWEs)
phrases = ["cell phone", "help desk", "data science", "artificial intelligence"]

# 1. Lowercasing
sentence = sentence.lower()

# 2. Replacing MWEs with underscores before tokenization
for phrase in phrases:
    sentence = sentence.replace(phrase, phrase.replace(" ", "_"))
    
# 3. Removing Punctuation
sentence = re.sub(r'[^\w\s]', '', sentence)

# 4. Tokenization
tokens = word_tokenize(sentence)

# 5. POS Tagging to filter only Nouns (NN, NNS, NNP, NNPS)
pos_tags = nltk.pos_tag(tokens)
nouns = [word for word, pos in pos_tags if pos.startswith('NN') or '_' in word]  # Keep MWEs regardless of POS

# 6. Removing Stop Words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in nouns if word not in stop_words]

# 7. Stemming (excluding MWEs)
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) if '_' not in word else word for word in filtered_tokens]

# 8. Reverting underscores to spaces for readability
final_tokens = [word.replace('_', ' ') for word in stemmed_tokens]

# Result
print("Original Tokens:", tokens)
print("POS Tagged:", pos_tags)
print("Filtered Nouns (including MWEs):", nouns)
print("Filtered Tokens (without stop words):", filtered_tokens)
print("Final Tokens:", final_tokens)