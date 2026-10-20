from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from collections import Counter

# Jednorazowa deklaracja tokenizera, stop words i stemmera
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
stem_cache = {}     # globalny cache dla wyników stemowania

def remove_stop_words(tokens):
    # Get English stopwords and tokenize
    return [word for word in tokens if word not in stop_words]

def stem_words(tokens):
    # Zoptymalizowany Algorytm Portera-Stemmera z buforowaniem (Cache)
    result = []
    for token in tokens:
        if token not in stem_cache:
            stem_cache[token] = ps.stem(token)
        result.append(stem_cache[token])
    return result
def vectorize_text(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = remove_stop_words(tokens)
    tokens = stem_words(tokens)

    return Counter(tokens)




# print(vectorize_text("This is an example sentence to provide a good demonstration / demonstrate"
#                      " the vectorization process. It includes stop words and will be stemmed."))