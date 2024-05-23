# query parsing function and it returns the term frequency for the given query
import glob
from stemming.porter2 import stem
import string
import re
import os

# Defining Rcv1Doc class with three attributes docID, terms and doc_len
class Rcv1Doc:

    # Constructor method to initialize the Rcv1Doc class
    def __init__(self, docID):
        self.docID = docID
        self.terms = {}
        self.doc_len = 0

    # method to get document id for a Rcv1Doc class
    def getDocId(self):
        return self.docID

    # method to get document length for a Rcv1Doc class
    def getDocLen(self):
        return self.doc_len

    # method to set document length for a Rcv1Doc class
    def setDocLen(self, doc_len):
        self.doc_len = doc_len

    # This method returns the sorted term list based on the frequency
    def get_term_list(self):
        # Sort terms based on frequency, with higher frequency terms first
        sorted_terms = sorted(self.terms.items(), key=lambda x: x[1], reverse=True)
        # Convert the sorted list of tuples back to a list of strings containing term and frequency
        sorted_terms_str = [f"{term}: {freq}" for term, freq in sorted_terms]
        return sorted_terms_str

    # This method adds the given term. It checks the current frequency of the term from the terms dictionary and,
    # if the term is not already in the dictionary, it returns 0 and add 1, else returns the current frequency and add 1
    def add_term(self, term):
        # self.doc_len += 1
        self.terms[term] = self.terms.get(term, 0) + 1

def parse_rcv1v2(stop_words, input_path):
    rcv1_collection = {}
    specific_doc = '71157.xml'
    file_path = os.path.join(input_path, specific_doc)
    
    if os.path.exists(file_path):
        start_end = False
        for line in open(file_path, 'r'):
            line = line.strip()
            if not start_end:
                if line.startswith("<newsitem "):
                    for part in line.split():
                        if part.startswith("itemid="):
                            docid = part.split("=")[1].split("\"")[1]
                            rcv1_doc = Rcv1Doc(docid)
                            break
                if line.startswith("<text>"):
                    start_end = True
            elif line.startswith("</text>"):
                break
            else:
                line = line.replace("<p>", "").replace("</p>", "")
                line = line.translate(str.maketrans('', '', string.digits)).translate(
                    str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                for word in line.split():
                    rcv1_doc.setDocLen(rcv1_doc.getDocLen() + 1)
                    term = stem(word.lower())
                    if len(term) > 2 and term not in stop_words:
                        rcv1_doc.add_term(term)
        rcv1_collection[rcv1_doc.getDocId()] = rcv1_doc

    return rcv1_collection

# This method is used to parse a data collection by giving input files path and the stop words list to eliminate
def parse_rcv1v2_old(stop_words, input_path):
    global rcv1_doc
    rcv1_collection = {}
    for file_ in glob.glob(input_path + "/*.xml"):  # using glob() to read all the matching .xml type files
        start_end = False  # to keep track of <text> </text> part of the document
        for line in open(file_):
            line = line.strip()
            # if start_end = False means, still we haven't found the <text> </text> part of the document
            # and this line could be used to find the itemid
            if (start_end == False):
                # if line.startswith("<newsitem "), then the docid is within that part and get the itemid
                if line.startswith("<newsitem "):
                    for part in line.split():
                        if part.startswith("itemid="):
                            docid = part.split("=")[1].split("\"")[1]  # get the docid by the itemid
                            # create a new Rcv1Doc object using it's constructor by passing docid
                            # and empty dictionary for terms and doc_len will be initialized to 0.
                            rcv1_doc = Rcv1Doc(docid)
                            break
                # if line.startswith("<text>") then, we found the <text> </text> part of the document,
                # and change the bool value of start_end = True
                if line.startswith("<text>"):
                    start_end = True
            elif line.startswith("</text>"):
                break  # Exit the for loop of processing each line of the file
            else:  # processing the inside of <text> </text> part of the document
                # replacing html tags (only <p> and </p> tags) within text content and not considering them as words
                line = line.replace("<p>", "").replace("</p>", "")
                line = line.translate(str.maketrans('', '', string.digits)).translate(
                    str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                # for all other words in the text considered to be added to the document length.
                for word in line.split():
                    # increasing the doc length for each word before applying stemming
                    rcv1_doc.setDocLen(rcv1_doc.getDocLen() + 1)
                    # Using porter2 stemmer to stem the term after converting it to lower case
                    term = stem(word.lower())

                    # terms which are not in the stop word list and terms length > 2 considered as terms and added to
                    # rcv1_doc terms dictionary
                    if len(term) > 2 and term not in stop_words:
                        rcv1_doc.add_term(term)

        # Add the Rcv1Doc object to the final collection
        rcv1_collection[rcv1_doc.getDocId()] = rcv1_doc
    return rcv1_collection

# This method returns a dictionary with {term:document_frequency} for all the terms in the documents with their
# document frequencies. ie. how many documents contains the term for a given document collection coll.
def my_df(coll):
    df = {}  # Initialize a dictionary to store document frequency (df) for each term
    # Iterate over each document in the collection
    for doc_id, rcv1_doc in coll.items():
        for term in rcv1_doc.terms.keys():
            try:
                df[term] += 1
            except KeyError:
                df[term] = 1
    return df

def get_input_path_and_stopword_list():
    stop_words_file = 'common-english-words.txt'  # Stop word file
    input_path = 'Data_Collection/'  # XML files path location

    # Open and read the given stop_words_file and store them into a list called 'stopword_list'
    stopwords_f = open(stop_words_file, 'r')
    stopword_list = stopwords_f.read().split(',')  # list of stop words
    stopwords_f.close()

    return input_path, stopword_list

# This method calculate and return the average document length of all documents in the given collection coll.
def avg_length(coll):
    total_doc_length = 0
    total_docs = len(coll)

    for rcv1_doc in coll.values():
        total_doc_length += rcv1_doc.getDocLen()

    avg_doc_length = total_doc_length / total_docs
    return avg_doc_length

# Parsing queries based on the model type and 
# Returns a dictionary of dictionry of {queryno: {query_term: freq}}
# For BM25 and JM_LM models query terms will be taken only query titles
# For PRM model, long-queries ie. query title, description and narratives will be taken
def parse_queries(query_file_path, stop_words, model_name):
    queries = {}

    with open(query_file_path, 'r') as f:
        q_lines = f.readlines()
        i = 0
        while i < len(q_lines):
            if q_lines[i].startswith('<num>'):
                query_num = q_lines[i].strip()[6:]

                j = i + 1
                while not q_lines[j].startswith('<title>'):
                    j += 1
                query_title = q_lines[j].strip()
                query_title = query_title.strip().replace("<title>", "").replace(
                            "</title>", "").strip()
                query_title = query_title.replace("<p>", "").replace("</p>", "")
                query_title = query_title.translate(str.maketrans('', '', string.digits)).translate(
                    str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                query_title = re.sub("\s+", " ", query_title)

                k = j + 1
                while not q_lines[k].startswith('<desc>'):
                    k += 1
                query_desc = ""
                while not q_lines[k].startswith('<narr>'):
                    query_desc += q_lines[k].strip()
                    k += 1
                query_desc = query_desc.strip().replace("<desc> Description:", "").replace(
                    "</desc> Description:", "").strip()
                query_desc = query_desc.replace("<p>", "").replace("</p>", "")
                query_desc = query_desc.translate(str.maketrans('', '', string.digits)).translate(
                    str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                query_desc = re.sub("\s+", " ", query_desc)

                m = k + 1
                query_narr = ""
                while not q_lines[m].startswith('<'):
                    line = q_lines[m].strip()
                    if line and line[0].islower():
                        query_narr += " "
                    query_narr += line
                    m += 1
                query_desc = query_desc.strip().replace("<narr> Narrative:", "").replace(
                    "</narr> Narrative:", "").strip()    
                query_narr = query_narr.replace("<p>", "").replace("</p>", "")
                query_narr = query_narr.translate(str.maketrans('', '', string.digits)).translate(
                    str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                query_narr = re.sub("\s+", " ", query_narr)

                if query_num not in queries:
                    queries[query_num] = {}

                if model_name == 'BM25' or model_name == 'JM_LM': 
                    sections = [query_title] # For BM25 and JM_LM models query terms will be taken only query titles
                elif model_name == 'PRM':
                    sections = [query_title, query_desc, query_narr] # For PRM model, long-queries ie. query title, description and narratives will be taken
                else:
                    raise ValueError("Invalid model type")

                for section in sections:
                    for term in section.split():
                        term = stem(term.lower())
                        if len(term) > 2 and term not in stop_words:
                            queries[query_num][term] = queries[query_num].get(term, 0) + 1
                
            i += 1
    #return a dictionary of dictionry of {queryno: {query_term: freq}}
    return queries