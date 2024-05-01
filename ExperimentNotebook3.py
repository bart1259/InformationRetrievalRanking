#!/usr/bin/env python
# coding: utf-8

# In[243]:


import pandas as pd
import os

if "DATASET_NAME" in os.environ:
    DATASET_NAME = os.environ["DATASET_NAME"]
else:
    DATASET_NAME= "news-aggregator"
    
print(f"Dataset name is: {DATASET_NAME}")

train_catalog_df = pd.read_csv(f"data/{DATASET_NAME}/train_catalog.csv")
train_queries_df = pd.read_csv(f"data/{DATASET_NAME}/train_queries.csv")
val_catalog_df = pd.read_csv(f"data/{DATASET_NAME}/val_catalog.csv")
val_queries_df = pd.read_csv(f"data/{DATASET_NAME}/val_queries.csv")
print(f"Loaded {len(train_catalog_df.index)} Documents")
print(f"Loaded {len(train_queries_df.index)} Judgments")


# In[244]:


# Data Augmentation

if "AUGMENTATION" in os.environ:
    AUGMENTATION = os.environ["AUGMENTATION"]
else:
    AUGMENTATION = "synonym"
    
print(f"Augmentation strategy is: {AUGMENTATION}")
    
if AUGMENTATION == "none":
    pass
elif AUGMENTATION == "synonym":

    import nlpaug.augmenter.word as naw

    aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.25)
    train_queries_df["input_text"] = train_queries_df["input_text"].apply(lambda x: aug.augment(x)[0])
    val_queries_df["input_text"] = val_queries_df["input_text"].apply(lambda x: aug.augment(x)[0])


# In[245]:


print(train_queries_df.head(20)["input_text"].values)


# In[150]:


import numpy.random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data.dataloader import DataLoader

class RandomRanker:
    def __init__(self):
        pass
    
    def train(self, catalog_df, queries_df):
        pass
    
    def prerun(self, catalog_df):
        pass
    
    def get_score(self, query, catalog_df):
        text = query["input_text"]
        return {
            "scores": np.random.uniform(0,1,size=len(catalog_df))
        }
    
def levenshtein_distance(word1, word2):
    if len(word1) < len(word2):
        return levenshtein_distance(word2, word1)

    if len(word2) == 0:
        return len(word1)

    previous_row = range(len(word2) + 1)

    for i, c1 in enumerate(word1):
        current_row = [i + 1]

        for j, c2 in enumerate(word2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]
    

def normalized_levenshtein_distance(word1, word2):
    distance = levenshtein_distance(word1, word2)
    max_length = max(len(word1), len(word2))
    return distance / max_length

    
class LevensteinRanker:
    def __init__(self):
        print("Wanring! This is a slow ranker")
        pass
    
    def train(self, catalog_df, queries_df):
        pass
    
    def prerun(self, catalog_df):
        pass
    
    def get_score(self, query, catalog_df):
        text = str(query["input_text"])
        
        return {
            "scores": catalog_df["text"].apply(lambda x: -normalized_levenshtein_distance(x, text)).values
        }
    
class BoWRanker:
    def __init__(self):
        self.vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', lowercase=True)

    def train(self, catalog_df, queries_df):
        self.vectorizer.fit(catalog_df['text'].str.lower())

    def prerun(self, catalog_df):
        self.bow_matrix = self.vectorizer.transform(catalog_df['text'].str.lower())
        
    def get_score(self, query, catalog_df):
        text = str(query["input_text"]).lower()
        query_vector = self.vectorizer.transform([text])
        scores = (self.bow_matrix * query_vector.T).toarray()
        return {
            "scores": scores.flatten()
        }
    
class TfidfRanker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b', lowercase=True)

    def train(self, catalog_df, queries_df):
        self.vectorizer.fit(catalog_df['text'].str.lower())

    def prerun(self, catalog_df):
        self.tfidf_matrix = self.vectorizer.transform(catalog_df['text'].str.lower())
        
    def get_score(self, query, catalog_df):
        text = str(query["input_text"]).lower()
        query_vector = self.vectorizer.transform([text])
        scores = (self.tfidf_matrix * query_vector.T).toarray()
        return {
            "scores": scores.flatten()
        }
    
class BM25Ranker:
    def __init__(self):
        pass
        
    def train(self, catalog_df, queries_df):
        pass
        
    def prerun(self, catalog_df):
        corpus = catalog_df['text'].str.lower().tolist()
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_score(self, query, catalog_df):
        text = str(query["input_text"]).lower()
        query_vector = text.split(" ")
        scores = self.bm25.get_scores(query_vector)
        return {
            "scores": scores
        }
    
class EmbeddingRanker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def train(self, catalog_df, queries_df, epochs=0):
        if epochs != 0:
            
            # Prepare the data for training
            examples = []
            for _, row in queries_df.iterrows():
                text = str(row['input_text'])
                positive_id = row['match_id']
                try:
                    positive_text = catalog_df.loc[catalog_df['catalog_id'] == positive_id, 'text'].values[0]
                    negative_ids = catalog_df.loc[catalog_df['catalog_id'] != positive_id, 'catalog_id'].sample(n=1).values
                    negative_text = catalog_df.loc[catalog_df['catalog_id'] == negative_ids[0], 'text'].values[0]
                    examples.append(InputExample(texts=[text, positive_text, negative_text]))
                except Exception as e:
                    pass

            print(f"{len(examples)} Examples Found")

            train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
            train_loss = losses.TripletLoss(self.model)

            self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, optimizer_params={'lr': 2e-6})

    def prerun(self, catalog_df):
        self.corpus = catalog_df['text'].str.lower().tolist()
        self.corpus_embeddings = self.get_embeddings(self.corpus)

    def get_score(self, query, catalog_df):
        query_embedding = self.get_embeddings([query["input_text"]])
        scores = cosine_similarity(query_embedding, self.corpus_embeddings)
        return {
            "scores": scores.flatten()
        }

    def get_embeddings(self, texts):
        return self.model.encode(texts)
    
embedding_ranker = EmbeddingRanker()
embedding_ranker.train(train_catalog_df, train_queries_df, epochs=0)
embedding_ranker.prerun(train_catalog_df)
embedding_ranker.get_score({"input_text": "Remote"}, train_catalog_df)
    
trained_embedding_ranker_1 = EmbeddingRanker()
trained_embedding_ranker_1.train(train_catalog_df, train_queries_df, epochs=1)
trained_embedding_ranker_1.prerun(train_catalog_df)
trained_embedding_ranker_1.get_score({"input_text": "Remote"}, train_catalog_df)

bow_ranker = BoWRanker()
bow_ranker.train(train_catalog_df, train_queries_df)
bow_ranker.prerun(train_catalog_df)
bow_ranker.get_score({"input_text": "Remote"}, train_catalog_df)

tf_idf_ranker = TfidfRanker()
tf_idf_ranker.train(train_catalog_df, train_queries_df)
tf_idf_ranker.prerun(train_catalog_df)
tf_idf_ranker.get_score({"input_text": "Remote"}, train_catalog_df)

bm_25_ranker = BM25Ranker()
bm_25_ranker.train(train_catalog_df, train_queries_df)
bm_25_ranker.prerun(train_catalog_df)
bm_25_ranker.get_score({"input_text": "Remote"}, train_catalog_df)


# In[151]:


from tqdm import tqdm

def evaluate(ranker, catalog_df, queries_df):
    ranks = []
    ranker.prerun(catalog_df)
    for i,row in tqdm(queries_df.iterrows(), total=len(queries_df.index)):
        input_query = dict(row)
        target_id = input_query["match_id"]
        judgment = input_query["judgment"]
        
        if judgment == True:
            del input_query["match_id"]
            del input_query["judgment"]
            
            scores = ranker.get_score(input_query, catalog_df)["scores"]
            sorted_catalog = catalog_df.iloc[np.argsort(-scores)]
            rank = np.where(sorted_catalog["catalog_id"].values == target_id)
            rank = rank[0][0] # FIXME: This could file if target_id is not in the catalog_df, in that case, skip
            ranks.append(rank)
          
    ranks = np.array(ranks)
    return {
        "ranks": ranks,
        "top_1": sum(ranks < 1) / len(ranks),
        "top_10": sum(ranks < 10) / len(ranks),
        "top_100": sum(ranks < 100) / len(ranks),
        "top_1000": sum(ranks < 1000) / len(ranks),
    }

report = ""

metrics = evaluate(RandomRanker(), val_catalog_df, val_queries_df)
report += (f'Random | Top 1: {metrics["top_1"]} | Top 10: {metrics["top_10"]} | Top 100: {metrics["top_100"]}\n')
metrics = evaluate(bow_ranker, val_catalog_df, val_queries_df)
report += (f'Bag of Words | Top 1: {metrics["top_1"]} | Top 10: {metrics["top_10"]} | Top 100: {metrics["top_100"]}\n')
metrics = evaluate(tf_idf_ranker, val_catalog_df, val_queries_df)
report += (f'TF-IDF | Top 1: {metrics["top_1"]} | Top 10: {metrics["top_10"]} | Top 100: {metrics["top_100"]}\n')
metrics = evaluate(bm_25_ranker, val_catalog_df, val_queries_df)
report += (f'BM25   | Top 1: {metrics["top_1"]} | Top 10: {metrics["top_10"]} | Top 100: {metrics["top_100"]}\n')
metrics = evaluate(embedding_ranker, val_catalog_df, val_queries_df)
report += (f'Sentance Transformer | Top 1: {metrics["top_1"]} | Top 10: {metrics["top_10"]} | Top 100: {metrics["top_100"]}\n')
metrics = evaluate(trained_embedding_ranker_1, val_catalog_df, val_queries_df)
report += (f'Fine Tuned Sentance Transformer | Top 1: {metrics["top_1"]} | Top 10: {metrics["top_10"]} | Top 100: {metrics["top_100"]}\n')
print(report)


# In[17]:


import time

# Save run
with open(f'outputs/report_{DATASET_NAME}_{AUGMENTATION}_{int(time.time())}.txt', 'w') as f:
    f.write(report)


# In[11]:


# def compare_rankers(ranker1, ranker2, catalog_df, queries_df, cutoff=10):
#     ranks = []
#     ranker1.prerun(catalog_df)
#     ranker2.prerun(catalog_df)
#     for i,row in tqdm(queries_df.iterrows(), total=len(queries_df.index)):
#         input_query = dict(row)
#         target_id = input_query["match_id"]
#         judgment = input_query["judgment"]
        
#         if judgment == True:
#             del input_query["match_id"]
#             del input_query["judgment"]
            
#             scores1 = ranker1.get_score(input_query, catalog_df)["scores"]
#             scores2 = ranker2.get_score(input_query, catalog_df)["scores"]
#             sorted_catalog1 = catalog_df.iloc[np.argsort(-scores1)]
#             sorted_catalog2 = catalog_df.iloc[np.argsort(-scores2)]
#             rank1 = np.where(sorted_catalog1["catalog_id"].values == target_id)
#             rank1 = rank1[0][0]
#             rank2 = np.where(sorted_catalog2["catalog_id"].values == target_id)
#             rank2 = rank2[0][0]
            
#             if rank1 < cutoff and rank2 > cutoff:
#                 print("Ranker 1 was better at matching \"" + input_query["input_text"] + "\" to \"" + catalog_df[catalog_df["catalog_id"] == target_id]["text"].values[0] + "\"")
#             if rank2 < cutoff and rank1 > cutoff:
#                 print("Ranker 2 was better at matching \"" + input_query["input_text"] + "\" to \"" + catalog_df[catalog_df["catalog_id"] == target_id]["text"].values[0] + "\"")
          
# compare_rankers(tf_idf_ranker, trained_embedding_ranker_1, val_catalog_df, val_queries_df)


# In[6]:


val_catalog_df


# In[1]:


# res = evaluate(bm_25_ranker, val_catalog_df, val_queries_df)


# In[3]:


# !pip install nlpaug


# In[ ]:




