from app import app
from flask import request, jsonify
import numpy as np
import pandas as pd
import gensim
import nltk
import string
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime
import asyncio
import aiofiles
import aiohttp
import jwt
import pandas as pd
import time
import requests
import numpy as np
from bs4 import BeautifulSoup
import json

## pk = public_key
## secret = private_key
## jwt.encode({'pk': 'payload'}, 'secret', algorithm='HS256')
encoded_jwt = jwt.encode({'pk': 'pk_KoukiYYjUNBNbkXfOv7Lk7gkfNBo5g'}, 'sk_LYtda2Eini8LCUr7yOGbeG0Pb2Geid', algorithm='HS256')
jwt = encoded_jwt.decode('utf-8')

import os
# __file__ refers to the file settings.py 
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')

content = np.load(os.path.join(APP_STATIC, 'content.npy'), allow_pickle=True)
slug_list = np.load(os.path.join(APP_STATIC, 'title.npy'), allow_pickle=True)

@app.route('/recommend')
def recommend():
    slug = request.args.get('slug')
    if (slug == None) :
        return "Slug is not defined"

    return jsonify(article_recommender(slug,
                   slug_list,
                   content)[2].tolist())


def give_recommendation(content,url_list,url,num_of_rec):
    months = 5
    document_matrix = np.array(content)
    AOI = np.array(content[url_list == url])
    rec_list = np.argsort(
        np.sum(
            np.abs(
                document_matrix - AOI
                )
            ,axis = 1)
    )
    filtered_list = rec_list[abs(extract_date_vect(url_list)[rec_list] - extract_date(url)) <= months]
    return(filtered_list[1:num_of_rec])

def article_sampling(slug_list,num_list,num_of_sample):
    idx = np.random.choice(num_list,num_of_sample,replace = False)
    print(idx)
    return(slug_list[idx])

def article_ranking(content,slug_list,num_list,num_of_sample):
    time = content[num_list,content.shape[1]-1]
    return(slug_list[num_list[np.argsort(time)][::-1][0:num_of_sample]])

# function to clean the articles
stopword = set(stopwords.words('english'))
stopword.add("'s")
stemmer = PorterStemmer()
def is_word(x):
    return len(re.findall('[0-9' + string.punctuation +']|nbsp', x)) == 0
def clean_articles(syllabus):
    # tokenizes it using nltk.word_tokenize
    result = word_tokenize(syllabus)
    
    # converts it to lower case
    result = [token.lower() for token in result]
    
    # removes stopwords that are present in the set from the cell above
    result = [token for token in result if not token in stopword]
    
    # uses stemmer to cut the word down to its stem
    result = [stemmer.stem(token) for token in result]
    
    # uses has_letter to remove words that don't have any letters
    result = [token for token in result if is_word(token) == True]
    
    # keeps only those words with length greater than 1 and less than 20.
    result = [token for token in result if ((len(token) > 2) and (len(token) < 20))]
    
    return result

try1 = gensim.models.LdaModel.load(os.path.join(APP_STATIC,"title_model.lda"))
try2 = gensim.models.LdaModel.load(os.path.join(APP_STATIC,"content_model.lda"))

dictionary_title = gensim.corpora.Dictionary.load(os.path.join(APP_STATIC,"title_dict"))
dictionary = gensim.corpora.Dictionary.load(os.path.join(APP_STATIC,"content_dict"))

def data_cleaning(new_data):
    try:
        title = clean_articles(new_data['title'])
    except:
        title = clean_articles(" ".join(new_data['slug'].split("-")))
    try:
        desc = clean_articles(new_data['content'])
    except:
        print(new_data['title_url'])
    bow_title = dictionary_title.doc2bow(title)
    bow_content = dictionary.doc2bow(desc)
    dummy_frame_t = pd.DataFrame(columns = list(np.arange(0,45)))
    title_probs = pd.concat([dummy_frame_t,t_output_probs(try1,bow_title)]).fillna(0)
    dummy_frame = pd.DataFrame(columns = list(np.arange(0,35)))
    content_probs = pd.concat([dummy_frame,output_probs(try2,bow_content)]).fillna(0)
    return([new_data.title_url],np.concatenate((content_probs,title_probs),
                         axis = None))

def output_probs(optimal_model,x):
    vector = optimal_model[x][0]
    base = pd.DataFrame(np.array(vector).T)
    base.columns = base.iloc[0,:]
    base = base.drop([0]).astype(float)
    return(base)

def t_output_probs(t_optimal_model,x):
    vector = t_optimal_model[x][0]
    base = pd.DataFrame(np.array(vector).T)
    base.columns = base.iloc[0,:]
    base = base.drop([0]).astype(float)
    return(base)

def extract_date(x):
    try:
        val = int(x[0:4])*12 + int(x[5:7])
    except:
        val = 0
    return(val)
extract_date_vect = np.vectorize(extract_date)

def give_recommendation(content,url_list,url,num_of_rec,months):
    document_matrix = np.array(content)
    AOI = np.array(content[url_list == url])
    rec_list = np.argsort(
        np.sum(
            np.abs(
                document_matrix - AOI
                )
            ,axis = 1)
    )
    filtered_list = rec_list[abs(extract_date_vect(url_list)[rec_list] - extract_date(url)) <= months]
    return(filtered_list[1:num_of_rec])

def article_sampling(url_list,num_list,num_of_sample):
    return(url_list[np.random.choice(num_list,num_of_sample,replace = False)])

def article_ranking(content,url_list,num_list,num_of_sample):
    content = np.array(content)
    time = content[num_list,content.shape[1]-1]
    return(url_list[num_list[np.argsort(time)][::-1][0:num_of_sample]])

def fetch_new(slug):
    #Make Request
    endpoint = 'https://dpn.ceo.getsnworks.com/v3/search?type=content&keywords=' + slug
    headers = {"Authorization": "Bearer " + jwt}
    r = requests.get(url = endpoint,headers = headers)
    item = r.json()['items'][0]
    
    #Empty Frame
    uuid = []
    ids = []
    titles = []
    content = []
    slugs = []
    published_dates = []
    abstract = []
    title_url = []
    
    #Append Frame
    uuid.append(item['uuid'])
    ids.append(item['id'])
    titles.append(item['title'])
    title_url.append(item['published_at'][0:4] + "/" + item['published_at'][5:7] + "/" + item['title_url'])
    slugs.append(item['slug'])
    soup = BeautifulSoup(item['content'], features="html.parser")
    for script in soup(['script','style']):
        script.decompose()
    content.append(soup.get_text().replace(u'\xa0',u' ').replace("\n"," "))
    published_dates.append(item['published_at'])
    abstract.append(item['abstract'])  
    
    #Assemble Frame
    articles_df = pd.DataFrame(data={'id' : ids,'uuid' : uuid,'title': titles, 'slug': slugs, 'content': content, 
                                 'published_date': published_dates,"title_url":title_url})
    
    return(articles_df)

def write_new(old_content,new_content,old_title,new_title):
    full_content = np.vstack([old_content, new_content])
    full_title = np.array(list(old_title) + list(new_title))
    np.save(os.path.join(APP_STATIC, 'content.npy'),full_content)
    np.save(os.path.join(APP_STATIC, 'title.npy'),full_title)
    return full_content,full_title

def article_recommender(article_title_url,added_url_list,added_comp):
    if np.any(added_url_list == article_title_url):
        return(added_comp,added_url_list,
               article_sampling(added_url_list,give_recommendation(added_comp,added_url_list,article_title_url,
                                                                   10,12),3))
    else:
        new_data = fetch_new(article_title_url[8:])
        new_title,new_content = data_cleaning(new_data.iloc[0,:])
        comp,url_list = write_new(added_comp, new_content, added_url_list, new_title)
        return(comp,url_list,
               article_sampling(url_list,give_recommendation(comp,url_list,article_title_url,
                                                             10,12),3))
    
  