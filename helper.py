import time
import random
from pymongo import MongoClient
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import sys
import operator
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from collections import Counter 
import numpy as np

def get_random(prefix,year,field="body",value={"$exists":1}):
    client = MongoClient()
    db = client.nyt
    col=db[prefix+str(year)]
    cursor=col.find({field:value})
    cant=cursor.count()
    print(cant)
    if cant>0:
        idx=random.randint(0,cant)
        out=cursor[idx]
    else:
        out=None
    return out

def count_words(prefix,fields,from_year,to_year,ngrams=10,min_tok=100):
    client = MongoClient()
    db = client.nyt
    col_voc=db["vocabulary_"+str(from_year)+"_"+str(to_year)]
    step=1000
    for year in range(from_year,to_year+1):
        print("Procesando el ano {}".format(year))
        col=db[prefix+str(year)]
        words={"_id":str(year),"n-grams":ngrams,"articles":col.find().count()}
        for n in range(1,(ngrams+1)):
                words[str(n)+"-grams"]=dict()
        
        for n in range(1,(ngrams+1)):
            print("Procesando los {}-gramas".format(n))
            start=time.time()
            cursor=col.find()
            cant=cursor.count()
            num=0
            for art in cursor:
                for field in fields:
                    for line in art[field]:
                        if len(line) > n:
                            for num_word in range(len(line)-n+1):
                                token=[line[idx] for idx in range(num_word,(num_word+n))]
                                if(n==1):
                                    token='-'.join(token)
                                    if token in words[str(n)+"-grams"]:
                                        words[str(n)+"-grams"][token]=words[str(n)+"-grams"][token]+1
                                    else:
                                        words[str(n)+"-grams"][token]=1
                                else:
                                    token_prev="-".join(token[:-1])
                                    if token_prev in words[str(n-1)+"-grams"]:
                                        token='-'.join(token)
                                        if token in words[str(n)+"-grams"]:
                                            words[str(n)+"-grams"][token]=words[str(n)+"-grams"][token]+1
                                        else:
                                            words[str(n)+"-grams"][token]=1                        
                num=num+1
                if num%step==0:
                    stop = time.time()
                    estim= (stop-start)*(cant-num)/num
                    sys.stdout.write("\r{} % Estimated time {} secs.".format(int(num/cant*100),int(estim)))
            print("\nEliminando tokens que aparecieron menos de {} veces...".format(min_tok))
            keys=list(words[str(n)+"-grams"].keys())
            for key in keys:
                if words[str(n)+"-grams"][key]<min_tok:
                    del words[str(n)+"-grams"][key]
            print("Quedaron {} tokens.".format(len(words[str(n)+"-grams"])))
        cant_words=0
        for n in range(1,(ngrams+1)):
            cant_words=cant_words+len(words[str(n)+"-grams"])
        words["size"]=cant_words
        col_voc.update({ "_id": str(year)},words, upsert=True)
        
def cant_articles(prefix,year):
    client = MongoClient()
    db = client.nyt 
    col=db[prefix+str(year)]
    cursor=col.find()
    cant=cursor.count()
    return cant

def get_vocabulary(voc_col,year):
    client = MongoClient()
    db = client.nyt
    col_voc=db[voc_col]
    voc=col_voc.find_one({"_id":str(year)})
    return voc

def compile_vocabulary(voc_col,name,ngrams=3,min_df=[0.001,0.01,0.05],max_df=[10,0.1,0.1]):
    client = MongoClient()
    db = client.nyt
    col_voc=db[voc_col]
    voc=col_voc.find_one({"_id":str(name)})
    for n in range(ngrams,0,-1):
        keys=list(voc[str(n)+"-grams"].keys())
        for key in keys:
            if voc[str(n)+"-grams"][key]<min_df[n-1]*voc["articles"]:
                del voc[str(n)+"-grams"][key]
            elif voc[str(n)+"-grams"][key]>max_df[n-1]*voc["articles"]:
                del voc[str(n)+"-grams"][key]
            else:
                text=key.split('-')
                tam=len(text)
                for k in range(1,n):
                    for num_word in range(0,(tam-k+1)):
                        aux_key=[text[idx] for idx in range(num_word,(num_word+k))]
                        aux_key='-'.join(aux_key)
                        if(aux_key=="in-the-united"):
                            print("holis")
                            print(str(k)+"-grams")
                        voc[str(k)+"-grams"][aux_key]=voc[str(k)+"-grams"][aux_key]-voc[str(n)+"-grams"][key]
        print("Quedaron {} tokens de {}-gramas".format(len(voc[str(n)+"-grams"]),n))
    cant_words=0
    for n in range(1,(ngrams+1)):
        cant_words=cant_words+len(voc[str(n)+"-grams"])
        voc["size"]=cant_words
    voc["_id"]="proc-"+name
    col_voc.update({ "_id": "proc-"+name},voc,upsert=True)
    print("Se salvo el vocabulario compilado en {}".format("proc-"+name))
    
def histo_voc(voc, size, plot_from=0, plot_to=100, keys=None,ngrams=3,name="histogram",plot=True): 
    tokens=dict()
    if keys==None:
        for n in range(1,ngrams+1):
            for key in voc[str(n)+"-grams"]:
                tokens[key]=voc[str(n)+"-grams"][key]
        tokens=dict(sorted(tokens.items(), key=operator.itemgetter(1),reverse=True))
    else:
        for key in keys:
            n=len(key.split('-'))
            if key in voc[str(n)+"-grams"].keys():
                tokens[key]=voc[str(n)+"-grams"][key]
            else:
                tokens[key]=0
    values=np.array(list(tokens.values()))
    values=values/sum(values)
    if plot:
        plt.plot(range(plot_from,plot_to), values[plot_from:plot_to],label=name)#, align='center',)
        plt.xticks(range(plot_from,plot_to), list(tokens.keys())[plot_from:plot_to],rotation=90)
    return list(tokens.keys())[:size],list(values)[:size]

def generate_dictionary(max_hist):
    count=dict()
    dictionary=dict()
    reverse_dictionary=dict()
    dictionary["UNK"]=0
    count["UNK"]=-1
    reverse_dictionary[0]="UNK"
    for idx,key in enumerate(max_hist):
        dictionary[key]=idx+1
        reverse_dictionary[idx+1]=key
    return dictionary, reverse_dictionary
        
def generate_dataset(file,prefix,year,dictionary,ngrams=3):
    import pickle
    data=list()
    client = MongoClient()
    db = client.nyt
    step=100
    start=time.time()
    col_arts=db[prefix+str(year)]
    cursor=col_arts.find({"body_norm":{"$exists":True}})
    cant=cursor.count()
    num=0
    for art in cursor:
        for line in art["body_norm"]:
            proc_line=line2idx(line,dictionary,ngrams)
            if(proc_line):
                data.append(proc_line)
        num +=1
        if num%step==0:
            stop = time.time()
            estim= (stop-start)*(cant-num)/num
            sys.stdout.write("\r year: {}. {} % Estimated time {} secs.".format(year,int(num/cant*100),int(estim)))
    with open(file+"-samples-"+str(year)+".pck","wb") as f:
        pickle.dump(data,f)
    
def line2idx(line, vocabulary,ngrams):
    proc_line=list()
    len_line=len(line)
    if len_line==0:
        return None
    eol=0
    word=0
    #print(line)
    proc_line.append(0)
    while eol==0:
        for n in range(ngrams,0,-1):
            if word+n<=len_line:
                token="-".join(line[word:word+n])
                if token in vocabulary:
                    proc_line.append(vocabulary[token])
                    word=word+n
                    break
            if n==1:
                proc_line.append(vocabulary["UNK"]) #UNK
                word=word+1
        if word>=len_line:
            eol=1
    #print(proc_line)
    return proc_line

def generate_epoch(from_year,to_year,batch_size,cant_batchs_per_year):
    cant_years=to_year-from_year+1
    cant_batchs=cant_batchs_per_year*cant_years
    batchs=np.arange(0,cant_batchs)
    random.shuffle(batchs)
    context=np.zeros(cant_batchs*batch_size)
    target=np.zeros(cant_batchs*batch_size)
    labels=np.zeros(cant_batchs*batch_size)
    years=np.zeros(cant_batchs)
    for idx,year in enumerate(range(from_year,to_year+1)):
        print()
        step=100000
        start=time.time()   
        cant=cant_batchs_per_year
        num=0
        couples_year=np.load("couples-"+str(year)+".npy")
        labels_year=np.load("labels-"+str(year)+".npy")
        indexes=labels_year.shape[0]
        for batch in range(0,cant_batchs_per_year):
            indexes_batch=random.sample(range(indexes),batch_size)
            context_batch=couples_year[indexes_batch,1]
            target_batch=couples_year[indexes_batch,0]
            labels_batch=labels_year[indexes_batch]
            start_idx=batchs[idx*cant_batchs_per_year+batch]
            start_idx_mult=start_idx*batch_size
            context[start_idx_mult:(start_idx_mult+batch_size)]=context_batch
            target[start_idx_mult:(start_idx_mult+batch_size)]=target_batch
            labels[start_idx_mult:(start_idx_mult+batch_size)]=labels_batch
            years[start_idx]=idx
            num=num+1
            if num%step==0:
                stop = time.time()
                estim= (stop-start)*(cant-num)/num
                sys.stdout.write("\r year: {}. {} % Estimated time {} secs.".format(year,int(num/cant*100),int(estim)))
    return target, context, labels, years
        
