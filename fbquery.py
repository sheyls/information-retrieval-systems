import utils
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
# fbquery means formatted boolean query. 
# This document takes care of the paranthesis and 
# the precedance for the boolean operators and paranthesis in the query.

def is_Lparanthesis(query_token):
    if query_token == "(":
        return True
    return False

def is_Rparanthesis(query_token):
    if query_token == ")":
        return True
    return False

def is_binaryoperator(query_token):
    if query_token == "&" or query_token == "|":
        return True
    return False

def convert(query_token):
    
    # Converts an infix query into postfix
    stack = []
    order = list()

    for token in query_token:
        if is_binaryoperator(token):
            while len(stack) and (preference(token) <= preference(stack[-1])):
                order.append(stack.pop())
            stack.append(token)
        
        elif is_Lparanthesis(token):
            stack.append(token)

        elif is_Rparanthesis(token):
            while len(stack) and stack[-1] != "(":
                order.append(stack[-1]) 
                stack.pop()
            if len(stack) and stack[-1] != "(":
                raise ValueError("Query is not formed correctly!")
            else:
                stack.pop()
                
        else:
            order.append(token)

    while len(stack):
        order.append(stack.pop())
            
    return order

def preference(query_token):
    preference_order = {"&": 1, "|": 0}
    if query_token == "&" or query_token == "|":
        return preference_order[query_token]
    return -1

def preprocess_bquery(data):
    data = utils.convert_lower_case(data)
    data = remove_punctuation_bquery(data) #remove comma seperately
    data = utils.remove_apostrophe(data)
    data = remove_stop_words_bquery(data)
    data = utils.stemming(data)
    data = utils.stemming(data) #needed again as we need to stem the words
    data = remove_punctuation_bquery(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words_bquery(data) #needed again as num2word is giving stop words 101 - one hundred and one
    
    # Transforming queries with no specified boolean operator to 'boolean queries'
    rquery = ""
    splited = data.split()
    for w in range(len(splited)):
        if w == len(splited) - 1:
            rquery = rquery + splited[w]
            break

        if(splited[w+1] in "&|~" or splited[w] in "&|~"):
            rquery = rquery + splited[w] + " "
        else:
            rquery = rquery + splited[w] + " " + "&" + " "
    return rquery

def remove_punctuation_bquery(data):
    symbols = "!\"#$%()*+./:;<=>?@[\]^_`{-}\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return str(data)

def remove_stop_words_bquery(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    first_word = True
        
    for i in range(len(words)):
        w= words[i]
        
        if(w=='or'):
            w='|'
            words[i]= w
        elif(w=='and'):
            w='&'
            words[i]= w
        elif(w=='no' or w=='not'):
            w='~'
            words[i]= w

        if w not in stop_words and len(w) >= 1:
            if(first_word):
                new_text= w
                first_word=False
            else:
                new_text = new_text + " " + w
    return new_text
    
