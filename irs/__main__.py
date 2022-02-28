import os
import sys
import re
import urllib.error
import http.client
import json
import ssl

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup


from googleapiclient.discovery import build
from config import DEVELOPER_KEY, SEARCH_ENGINE_ID
from irs import rocchioAlgo as ralgo
from irs import index

def main():
    query = sys.argv[1]
    reqPrecision = float(sys.argv[2])
    if reqPrecision > 1 or reqPrecision <0:
        print('Precision value must be between 0 and 1')
        return
    currentPrecision = 0
    indexers = {zone: index.Vectorize(zone) for zone in ['title', 'summary', 'content']}
    queryAdder = ralgo.RocchioClass()
    while (currentPrecision < reqPrecision):
        print('Your Query details are: \n')
        print('Query is: ', query)
        print('\n Precision needed: ', reqPrecision)
        print('\n')       
        googleResults = googleQueryAPI(query)
        if len(googleResults) < 10:
            print('Results are not sufficient. Terminating the query.\n')
            break
        requestFeedback(googleResults)
        addContent(query, googleResults)
        relevant = [result['id'] for result in googleResults if result['relevant']]
        nonRelevant = [result['id'] for result in googleResults if not result['relevant']] 
        if googleResults:
            currentPrecision = len(relevant)/len(googleResults)
        else:
            currentPrecision = 0
            print('Precision is 0, terminating the search.')
            break
        query = rmvPunct(query)
        zoneThreads = []
        for zone in indexers:
            indexers[zone].reset()
            indexers[zone].buildReqVectors(googleResults, query)       
        print('\nAchieved precision: ', currentPrecision)
        query = queryAdder.rocchioAlgorithm(query, indexers, relevant, nonRelevant)

def requestFeedback(results):
    """Request feedback for each result"""
    print('Google Results are:\n')
    for i, result in enumerate(results):
        print('Result ', i+1)
        print('*** \n')
        print('Title: ', result['title'])
        print('URL: ', result['url'])
        print('Summary: ', result['summary'])
        print('*** \n')
        isRelevant = input("Relevant (Y/N)?")
        if not re.match("^[Y,y,N,n]{1,1}$", isRelevant):
            print('Please type in Y or N (or y or n)')
            isRelevant = input("Relevant (Y/N)?")
        result.update({'relevant': True if isRelevant.upper() == "Y" else False})
    return results

def rmvPunct(text):
    ''' Remove unnecessary punctuations'''
    wList = re.split('\W+', text)
    while '' in wList:
        wList.remove('')
    wList = [word.lower() for word in wList]
    return wList

def googleQueryAPI(query):
    service = build("customsearch", "v1", developerKey=DEVELOPER_KEY)
    res = service.cse().list(
        q=query,
        cx=SEARCH_ENGINE_ID,
    ).execute()   
    if 'items' in res.keys():
        results = res['items']
        rfmat = [{'id': idx, 'title': result['title'], 'url': result['link'], 'summary': result['snippet']} for idx, result in enumerate(results)]
        return rfmat
    else:
        return []
       
def addContent(query, documents):
    for doc in documents:
        text = ""
        url = doc['url']
        if(url.find("pdf")==-1):
            try:
                html_page = urlopen(url).read()
                textBeautify = BeautifulSoup(html_page, 'html5lib')
                data = textBeautify.findAll('p')
                data = [p.get_text().replace('\n', '').replace('\t','') for p in data]
                if data:
                    text = " ".join(data)
                else:
                    text = ""
            except (http.client.IncompleteRead, http.client.RemoteDisconnected, urllib.error.URLError, ssl.CertificateError):
                text = ""
        doc.update({'content': text})

       
if __name__ == '__main__':
    main()
