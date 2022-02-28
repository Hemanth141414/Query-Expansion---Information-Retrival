import math
import numpy as np

class RocchioClass:

    def __init__(self):
        self.aplha = 1.0
        self.beta = 0.75
        self.gamma = 0.15
        #Below are the zone constants or contributions to the presence of the next potential word to add to query.
        self.zone_title_we = 0.6
        self.zone_summ_we = 0.25
        self.zone_content_we = 0.15
        self.docVector = None
        self.queryVector = None
        self.rocchioVector = None

    def reset(self):
        self.__init__()

    def calculateDocumentWeight(self, indexers):
        #totalWords is same for all the zones as this is total unique words collected.
        totalWords = indexers['content'].totalWords
        totalDocuments = indexers['content'].totalDocuments
        self.docVector = np.zeros((totalDocuments, totalWords))
        docZoneWeights = dict()
        for zone in ['title', 'summary', 'content']: 
            docZoneWeights[zone] = np.zeros((totalDocuments, totalWords))
            for docId in range(totalDocuments):
                for termIdx in range(totalWords):
                    tfIdf = indexers[zone].scaledTermFreq(termIdx, docId) * indexers[zone].invDocFreq(termIdx)
                    docZoneWeights[zone][docId][termIdx] = docZoneWeights[zone][docId][termIdx] + tfIdf 
                normaliseVal = np.linalg.norm(docZoneWeights[zone][docId])    
                if normaliseVal != 0:
                    docZoneWeights[zone][docId] = docZoneWeights[zone][docId]/normaliseVal
        tmp = self.zone_title_we * docZoneWeights['title'] + self.zone_summ_we * docZoneWeights['summary']
        self.docVector = tmp + self.zone_content_we * docZoneWeights['content']
        
    def calculateQueryWeight(self, index, query):
        self.queryVector = np.zeros(index.totalWords)
        
        for term in query:
            termIdx = index.getTermIdx(term)
            self.queryVector[termIdx] = math.log(1 + query.count(term), 10) * index.invDocFreq(termIdx)
            if np.linalg.norm(self.queryVector) != 0:
                self.queryVector /= np.linalg.norm(self.queryVector)


    def rocchioAlgorithm(self, query, indexers, relevant, nonRelevant):
        self.calculateDocumentWeight(indexers)
        self.calculateQueryWeight(indexers['content'], query)
        self.rocchioVector = self.aplha * self.queryVector
        for doc_id in relevant:
            self.relevantDocWeight=(self.beta/len(relevant)) * self.docVector[doc_id]
            self.rocchioVector = self.rocchioVector + self.relevantDocWeight
        for doc_id in nonRelevant:
            self.nonRelevantDocWeight=(self.gamma/len(nonRelevant)) * self.docVector[doc_id]
            self.rocchioVector = self.rocchioVector - self.nonRelevantDocWeight
        self.rocchioVector = self.rocchioVector.clip(min=0)
        #words are ranked in ascending order here.
        wordIdsRanked = list(np.argsort(self.rocchioVector))        
        for term in query:
            termIdx = indexers['content'].getTermIdx(term)
            if termIdx in wordIdsRanked:
                wordIdsRanked.remove(termIdx)
        queryS = " ".join(query)        
        #Adding the two new words, popping the last two words as the words are in ascending order. 
        for termIdx in wordIdsRanked[-2:]:
            queryS = queryS + " " + indexers['content'].getTerm(termIdx)
        return queryS
