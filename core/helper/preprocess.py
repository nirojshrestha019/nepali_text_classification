import string
import snowballstemmer
import nltk


class PreProcess:
    def __init__(self):
        self.nepali_punctuation = "।,‘’!"
        self.nepali_counting = "०१२३४५६७८९"
        self.nepali_stopWords = set(nltk.corpus.stopwords.words('nepali'))
        self.stemmer = snowballstemmer.stemmer("nepali")

    def start(self, dataframe):
        dataframe['data'] = dataframe['data'].replace({r'\r': ' ', r'\n': ' '}, regex=True)
        dataframe['data'] = dataframe['data'].apply(lambda x: ' '.join(
            x.replace(",", "").replace("‘", "").replace("’", "") for x in x.split() if
            x not in string.punctuation and x not in self.nepali_punctuation))
        dataframe['data'] = dataframe['data'].apply(lambda x: ' '.join(
            x for x in x.split() if not any(each_nepali_count in x for each_nepali_count in self.nepali_counting)))
        dataframe['data'] = dataframe['data'].apply(
            lambda x: ' '.join(x for x in x.split() if not x in self.nepali_stopWords))
        dataframe['data'] = dataframe['data'].apply(
            lambda x: " ".join([self.stemmer.stemWord(word) for word in x.split()]))
        return dataframe
