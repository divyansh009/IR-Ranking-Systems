#Importing all the necessary packages required for pre-processing:
import numpy as np #Used for array operations
import nltk #Used as a basic package for nlp operations
from nltk.corpus import stopwords #Helps in stop words removal
from nltk.stem import WordNetLemmatizer #Helps in lemmatization process
from nltk.stem import PorterStemmer #Helps in stemming process
#Helps in tokenization of words:
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize
import math #Used for mathematical equations solving
import pickle #Used to dump files to memory so, that we don't need to train model again anad again for a long time.
import os #Used to iterate files in the local system
import glob #Used for identifying paths in local system
import re #Used for removal of non-ascii characters
import sys #Used to manipulate different parts of the Python runtime environment
from pathlib import Path #Makes it very easy and efficient to deal with file paths
from collections import Counter #Used for carrying out frequency count
import pandas as pd #For working with dataframes
stop=set(stopwords.words('english')) #Storing all stop words in a set data structure
ps=PorterStemmer() #Creating object of PorterStemmer
#Opening pickle file for posting_list and saving it as a variable:
with open('All/Saved/posting_list.pkl',"rb") as temp:
    posting_lists=pickle.load(temp)
    temp.close()
#Opening pickle file for file_idx and saving it as a variable:
with open('All/Saved/file_idx.pkl',"rb") as temp:
    file_idx=pickle.load(temp)
    temp.close()
#Opening pickle file for df and saving it as a variable:
with open('All/Saved/df.pkl','rb') as file:
    DF=pickle.load(file)
    file.close()  
#Opening pickle file for doc_words and saving it as a variable:
with open('All/Saved/doc_words.pkl','rb') as file:
    doc_words=pickle.load(file)
    file.close()   
#Opening pickle file for doc_norm and saving it as a variable: 
with open('All/Saved/doc_norm.pkl','rb') as file:
    doc_norm=pickle.load(file)
    file.close()
#Opening pickle file for doc_len and saving it as a variable:
with open('All/Saved/doc_len.pkl','rb') as file:
    doc_len=pickle.load(file)
    file.close()
#We'll form a set of posting list keys so that we get all unique words:
unique_words=set(posting_lists.keys())
print(len(unique_words))#Printing total number of unique words we have
#Now, we'll create various functions that can be used later:
def remove_non_ascii_characters(data):
    pattern=re.compile('[^a-zA-Z0-9\s]')
    out=re.sub(pattern,'',data)
    return out
#Defining a function for boolean retrieval system 
def BRS(query,counter=5): #Keeping the value of counter as 5 according to question's requirements
    text=remove_non_ascii_characters(query) #Text cleaning by removing non ascii characters first
    text=re.sub(re.compile('\d'),'',text) #Removal of single digits
    words=word_tokenize(text) #Performing word tokenization now
    words=[word.lower() for word in words] #Performing lower casing of all characters
    words=[ps.stem(word) for word in words] #Stemming the words to avoid redundancy
    words=[word for word in words if word not in stop] #Stop words removal
    if len(words)>1: 
        query_words=[words[0]]
    else:
        query_words=words 
    for i in range(1,len(words)): #
        if words[i] not in ["and","or"]: #If and or OR is not present in query, then append AND in boolean retrieval model:
            if query_words[-1] not in ["and","or"]:
                query_words.append("and")
                query_words.append(words[i])
            else:
                query_words.append(words[i])
        elif query_words[-1] not in ["and","or"]:
            query_words.append(words[i])
    operators=[]
    main_words=[]
    for w in query_words: #Iterating query words
        if w.lower() in ["and","or"]:
            operators.append(w.lower())
        else:
            main_words.append(w)
    n=len(file_idx)
    word_vector=[]
    word_vector_matrix=[]
    for w in main_words:
        word_vector=[0]*n
        if w in unique_words:
            for x in posting_lists[w].keys():
                word_vector[x]=1
        word_vector_matrix.append(word_vector)
    for w in operators:
        vector1=word_vector_matrix[0]
        vector2=word_vector_matrix[1]
        if w=="and":
            result=[b1&b2 for b1,b2 in zip(vector1,vector2)]
            word_vector_matrix.pop(0)
            word_vector_matrix.pop(0)
            word_vector_matrix.insert(0,result)
        else:
            result=[b1|b2 for b1,b2 in zip(vector1,vector2)]
            word_vector_matrix.pop(0)
            word_vector_matrix.pop(0)
            word_vector_matrix.insert(0,result)
    final_word_vector=word_vector_matrix[0]
    cnt=0
    files=[]
    for i in final_word_vector:
        if i==1:
            files.append(file_idx[cnt])
            counter-=1
        cnt+=1
        if counter==0:
            break
    return files
#Defining a function for TF-IDF
def TFIDF(query,counter=5):
    text=remove_non_ascii_characters(query)
    text=re.sub(re.compile('\d'),'',text)
    words=word_tokenize(text)
    words=[word.lower() for word in words]
    words=[ps.stem(word) for word in words]
    words=[word for word in words if word not in stop]
    words=[word for word in words if word in DF.keys()]
    q=[]
    q_norm=0
    for w in words:
        tf_idf=(words.count(w)*math.log(len(file_idx)/DF[w]))
        q.append(tf_idf)
        q_norm+=tf_idf**2
    q_norm=math.sqrt(q_norm)
    q=np.array(q)/q_norm
    score={}
    for i in range(len(file_idx)):
        doc_v=[]
        for w in words:
            tf_idf=(doc_words[i].count(w)*math.log(len(file_idx)/DF[w]))
            doc_v.append(tf_idf)
        doc_v=np.array(doc_v)/doc_norm[i]
        score[i]=np.dot(q,doc_v)
    score=sorted(score.items(),key=lambda x:x[1],reverse=True)
    files=[]
    for i in score:
        counter-=1
        files.append([file_idx[i[0]],i[1]])
        if counter==0:
            break
    return files
k=0
Ld=doc_len
N=len(file_idx)
for i in Ld:
    k+=Ld[i]
Lavg=k/N
#Defining a function for calculating IDF that will be useful in BM25 model later
def IDF(q):
    DF1=0
    if q in DF:
        DF1=DF[q]
    ans=math.log((N-DF1+0.5)/(DF1+0.5))
    return ans
#Defining a function for score that will be useful in BM25 model later
def score_doc(q,score):
    q=remove_non_ascii_characters(q)
    q=re.sub(re.compile('\d'),'',q)
    words=word_tokenize(q)
    words=[word.lower() for word in words]
    words=[ps.stem(word) for word in words]
    words=[word for word in words if word not in stop]
    for i in range(len(file_idx)):
        score[i]=0
        for qi in words:
            TF=0
            if qi in posting_lists:
                if i in posting_lists[qi]:
                    TF=posting_lists[qi][i]
            idf=IDF(qi)
            ans=idf*(k+1)*TF/(TF+k*(1-b+b*(Ld[i]/Lavg)))
            score[i]+=ans
#We will now take the ideal values for variables b and k that will be used in BM25:
b=0.75
k=1.2
#Defining a function for BM25:
def BM25(query,counter=5):
    score={}
    for i in range(len(file_idx)):
        score[i]=0
    score_doc(query,score)
    score=sorted(score.items(),key=lambda item: item[1],reverse=True)
    count=counter
    files=[]
    for i in score:
        count-=1
        files.append([file_idx[i[0]],i[1]])
        if count==0:
            break
    return files
#NOw, we'll take input from user in QRel format:
query_list=pd.read_csv(sys.argv[1],sep='\t',header=None)
query_list.columns=['qid','query']
csv=[]
for index, row in query_list.iterrows(): #We'll iterate the queries one by one using for loop
    files=BRS(row['query']) #Applying Boolean Retrieval System on our query first.
    for file in files:
        csv.append([row['qid'],1,file,1]) #Appending the output recieved from the model in csv file
    if len(files)<5: #Handling case if the total number of output files are less than 5
        remaining=[i for i in file_idx.values() if i not in files] #Iterate the non relevant files and store it in an array
        remaining=remaining[:5-len(files)]
        for file in remaining:
            csv.append([row['qid'],1,file,0]) #Store it in output file and mark as relevance as 0
#Now, we'll save the file as csv in Output_files folder
pd.DataFrame(csv,columns=['QueryID','Iteration','DocId','Relevance']).to_csv('Output_files/BRS.csv',index=False) 
csv=[] #Emptying array of csv again to apply for next model
for index, row in query_list.iterrows(): #We'll iterate the queries one by one again using for loop
    files=TFIDF(row['query']) #Applying TF-IDF System on our query first.
    for file in files: #Iterating all files received by the model
        relevance=0 #Initializing relevance as 0
        if file[1]>0: #If file[1] is not 0, then it's relevant. Then we'll change it's relevance to 1.
            relevance=1
        csv.append([row['qid'],1,file[0],relevance]) #Appending to array
#Now, we'll save the file as csv in Output_files folder
pd.DataFrame(csv,columns=['QueryID','Iteration','DocId','Relevance']).to_csv('Output_files/TFIDF.csv',index=False)
csv=[] #Emptying array of csv again to apply for next model
for index, row in query_list.iterrows(): #We'll iterate the queries one by one again using for loop
    files=BM25(row['query']) #Applying BM25 System on our query first.
    for file in files: #Iterating all files received by the model
        relevance=0 #Initializing relevance as 0
        if file[1]>0: #If file[1] is not 0, then it's relevant. Then we'll change it's relevance to 1.
            relevance=1
        csv.append([row['qid'],1,file[0],relevance]) #Appending to array
#Now, we'll save the file as csv in Output_files folder
pd.DataFrame(csv,columns=['QueryID','Iteration','DocId','Relevance']).to_csv('Output_files/BM25.csv',index=False)
