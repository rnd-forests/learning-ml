import nltk
import string
from collections import Counter


def not_stopwords(text):
    stopwords = nltk.corpus.stopwords.words("english")
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)


# Tokenizing the given text
def get_tokens():
    with open('./1.txt') as file:
        tokens = nltk.word_tokenize(file.read().lower().translate(string.punctuation))
        return tokens

if __name__ == "__main__":
    tokens = get_tokens()
    print("tokens[:20] = {}", tokens[:20])

    count1 = Counter(tokens)
    print("before: len(count1) = {}", len(count1))

    filtered1 = [w for w in tokens if not w in nltk.corpus.stopwords.words('english')]
    print("filtered1 tokens[:20] = {}", filtered1[:20])

    count1 = Counter(filtered1)
    print("after: len(count1) = {}", len(count1))
    print("most_common = {}", count1.most_common(10))

    # Tag each token with its associated type (Adverb, adjective, noun, etc.)
    # Stackoverflow: https://stackoverflow.com/questions/1833252/java-stanford-nlp-part-of-speech-labels
    tagged1 = nltk.pos_tag(filtered1)
    print("tagged[:20] = {}", tagged1[:20])
