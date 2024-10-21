from wordhoard import Synonyms

# Download WordNet data if not already downloaded
# nltk.download('wordnet')


from wordhoard import Hypernyms

hypernym = Hypernyms(search_string='train')
hypernym_results = hypernym.find_hypernyms()
print("locomotive" in hypernym_results)

synonym = Synonyms(search_string='locomotive')
synonym_results = synonym.find_synonyms()
print("train" in synonym_results)

# def are_synonyms(word1, word2):
#     # Get the synsets for each word
#     synsets_word1 = wn.synsets(word1)
#     synsets_word2 = wn.synsets(word2)
    
#     # Check if there is any intersection between the synsets of both words
#     for synset1 in synsets_word1:
#         for synset2 in synsets_word2:
#             if synset1 == synset2:
#                 return True
#     return False

# def is_hyponym(word1, word2):
#     # Get the synsets (possible meanings) of each word
#     synsets_word1 = wn.synsets(word1)
#     synsets_word2 = wn.synsets(word2)
    
#     # For each synset of word1 (possible meanings of word1)
#     for synset1 in synsets_word1:
#         # For each synset of word2 (possible meanings of word2)
#         for synset2 in synsets_word2:
#             # Check if synset1 is a hyponym of synset2 (i.e., word1 is a type of word2)
#             if synset2 in synset1.hypernyms():
#                 return True
#     return False

# print(are_synonyms("locomotive", "train"))