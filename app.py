from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from langdetect import detect
import numpy as np
import gradio as gr
import re
import torch
import requests
import os



summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
vectorizer = TfidfVectorizer(stop_words='english')
API_KEY=os.getenv("API_KEY")

def text_fetch(url, driver):
    try:
        print(url)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        cookie_selectors = [
            'div.cookie-banner',
            'div.cookie-popup',
            'div.cookie-notice',
            'footer',
            'aside',
            'div#cookie-consent',
            'div#cookie-banner',
            'nav',
            'header']
        for selector in cookie_selectors:
            for element in soup.select(selector):
                element.decompose()
        para = soup.find_all('p')
        cleaned_text=''
        despose={'cookies','cookie','news','privacy',"\n", "verifying you are human"}
        for p in para:
            try:
                text=p.get_text()
                k=text
                k.lower()
                l=set(k)
                if l.intersection(despose):
                    continue
                cleaned_text+=' '+text
            except:
                continue
        print("ok")
        return cleaned_text
    except TimeoutException:
        return None
    except WebDriverException as e:
        return None
    except Exception as e:
        return None


def api_search(query):
    params = {
        'access_key': API_KEY,
        'query': query,
        'type': 'news',
        'auto_location': 0,
        'gl': 'in',
        'hl': 'en'
    }
    api_result = requests.get('https://api.serpstack.com/search', params)
    api_response = api_result.json()
    anchors=[]
    for news in api_response['news_results']:
        anchors.append(news['url'])
    print(len(anchors))
    return anchors

def clean_text(text):
    try:
        if detect(text) == 'en':
            cleaned_text = re.sub(r'[^a-zA-Z\s.,?!]', '', text)
            return cleaned_text
        else:
            return ''
    except Exception as e:
        return ''


def text_generation(query):
    links = api_search(query)
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(10)
    print("driver started")
    pair = {}
    i=1
    for url in links:
        print(i,end=' ')
        i+=1
        text = text_fetch(url,driver)
        if text:
            if url[-1]=='/':
                url=url[:-1]
            text=clean_text(text)
            pair[text] = url
    driver.quit()
    print("driver quit")
    print(len(pair),"pairs")
    return pair


def clustering(pairs,e=0.8):
    texts=[text for text in pairs]
    try:
        X = vectorizer.fit_transform(texts)
        
        cosine_sim_matrix = cosine_similarity(X)
        cosine_dist_matrix = 1 - cosine_sim_matrix
        cosine_dist_matrix = np.clip(cosine_dist_matrix, 0.0, 1.0)
        db = DBSCAN(metric="precomputed", eps=e, min_samples=2)
        labels = db.fit_predict(cosine_dist_matrix)
    except:
        return []
    best_l=-1
    best_cluster=None
    cluster={}
    for i, label in enumerate(labels):
        if label<0:
            continue
        if label not in cluster:
            cluster[label]=[[],[]]
        cluster[label][0].append(texts[i])
        cluster[label][1].append(pairs[texts[i]])
    best_l=len(cluster)
    best_cluster=cluster
    if len(cluster)<3:
        cluster=clustering(pairs,e-0.1)
    for i in cluster:
        cluster[i][0].sort(key = lambda x:len(x))
    if best_l>len(cluster):
        return best_cluster
    return cluster


def summarization(text):
    text=text.split()
    if len(text)<=250:
        return None
    text=' '.join(text[0:min(700,len(text))])
    summary = summarizer(text, max_length=250, min_length=100, do_sample=False)
    return summary[0]['summary_text']


def output(cluster):
    label=0
    out={}
    while label>-1:
        if label not in cluster:
            break
        summary = summarization(cluster[label][0][-1])
        if not summary:
            continue
        out[summary]=cluster[label][1]
        label+=1
    return out


def query(company,domain=''):
    query = company+' '+domain
    pairs = text_generation(query)
    print("clustering start")
    cluster = clustering(pairs)
    print(len(cluster),"clusters")
    print("summarization")
    result = output(cluster)
    print("Done")
    return result



def gradio_fun(name,domain=''):
    result = query(name,domain)
    out_str = f"\"{name.capitalize()}\" Report on \"{domain.capitalize()}\" domain.\n\n"
    i=1
    for text in result:
        out_str+=str(i)+". "+text+"\n\nSupporting URLs\n"
        i+=1
        for link in result[text]:
            out_str+=">>"+link+"\n"
        out_str+='\n\n'
    return out_str




with gr.Blocks(fill_height=True) as demo:
    title="Company Insight",
    description="Fill below information to see Insight."
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(placeholder="Organization Name",label='Organization')
            domain = gr.Textbox(placeholder="Domain",label='Domain')
            submit = gr.Button(value="Submit")
        with gr.Column():
            result = gr.Textbox(label="Output",lines=28)
    
    submit.click(gradio_fun, inputs=[name,domain], outputs=result)
demo.launch()
