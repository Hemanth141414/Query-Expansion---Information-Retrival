import re
import math
import re
import nltk
import numpy as np

STOPWORDS = open('stopwords.txt').read().split('\n')

class Vectorize:
    def __init__(self, zone='content'):
        '''Intialize objects of this class with default values'''
        self.zone = zone
        self.totalDocuments = 0
        self.totalWords = 0
        self.termToIdx = dict()
        self.rev_mapper = list()
        self.docs_wrd_freq_vecs = list()
        self.docs_weights_vectors = list()
    
    def reset(self):
        self.__init__(self.zone)
    
    def getTerm(self, termToIdx):
        '''get term/word from index method'''
        for wrd in self.termToIdx:
            if self.termToIdx[wrd] == termToIdx:
                return wrd
    
    def getTermIdx(self, wrd):
        '''get term/word index from term/word method'''
        return self.termToIdx[wrd]   

    def scaledTermFreq(self, wrd_idx, docId):
        '''This is log normalised term frequency which is a part of recommended tf-idf scheme'''
        return math.log(1 + self.docs_wrd_freq_vecs[docId][wrd_idx], 10)

    def invDocFreq(self, wrd_idx):
        '''This is inverse document frequnecy which is a part of recommended tf-idf scheme'''
        if self.rev_mapper[wrd_idx]:
            return math.log(self.totalDocuments/len(self.rev_mapper[wrd_idx]))
        else:
            return 0
    
    def preparation(self, documents, query):
        ''' Prepration before calculating the tf-idf weighting'''
        documents_wrds = []
        wordList = []
        wordList += query
        for document in documents:
            totalWords = ' '.join([document[zone] for zone in ['title', 'summary', 'content']])
            totalWords = self.rmvPunct(totalWords)
            totalWords = self.removeStopwords(totalWords, retain=query)
            zone_wrds = document[self.zone]
            zone_wrds = self.rmvPunct(zone_wrds)
            zone_wrds = self.removeStopwords(zone_wrds, retain=query)
            documents_wrds.append(zone_wrds)
            wordList += totalWords
        uniQWrdList = list(set(wordList))
        return uniQWrdList, documents_wrds
    
    
    def buildReqVectors(self, documents, query):
        '''Build required vectors, inverse term-doc and idf frequency'''
        self.totalDocuments = len(documents)
        uniQWrdList, documents_wrds = self.preparation(documents, query)
        self.totalWords = len(uniQWrdList)
        self.docs_wrd_freq_vecs = np.zeros((self.totalDocuments, self.totalWords))
        self.termToIdx = {wrd: idx for idx, wrd in enumerate(uniQWrdList)}
        self.rev_mapper = [set() for _ in range(len(uniQWrdList))]
        for docId, wrds in enumerate(documents_wrds):
            for wrd in wrds:
                wrd_idx = self.getTermIdx(wrd)
                self.rev_mapper[wrd_idx].add(docId)
                self.docs_wrd_freq_vecs[docId][wrd_idx] = 1 + self.docs_wrd_freq_vecs[docId][wrd_idx]
    
    def rmvPunct(self, text):
        ''' Remove unnecessary punctuations'''
        wList = re.split('\W+', text)
        while '' in wList:
            wList.remove('') 
        wList = [word.lower() for word in wList]
        return wList    
    
    def removeStopwords(self, wList, retain=[]):
        '''Remove the stop words'''
        if retain and isinstance(retain, str):
            retain = split_remove_punctuation(retain)
        for w in STOPWORDS:
            if w not in retain:
                while w in wList:
                    wList.remove(w)
        return wList
