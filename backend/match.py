import pandas as pd
import numpy as np
import networkx as nx;
import os

from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()


csv_path = os.getenv("CSV_PATH")
api_key = os.getenv("OPENAI_API_KEY")
max_godchildren = int(os.getenv("MAX_GODCHILDREN"))

# Área com configuráveis
remove_idx = [0]
metadata_idx = [1,2,3]
weight_idx = [11]
# Colunas não usadas, uma tem dados gerais, a de peso está desatualizada,
# mas vai ser usada para o proprio usuário manipular os pesos
# atualmente o ajuste é manual

binary_idx = [4,8]

email_col_name = 'Nome_de_usuário'

df = pd.read_csv(csv_path)
df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)

ST_embedder = SentenceTransformer('all-mpnet-base-v2')
OAI_embedder = OpenAIEmbeddings(api_key=api_key, model='text-embedding-3-large')

# Utiliza os valores configurados para ignorar os dados que não são relevantes
remove_cols = [df.columns[i] for i in remove_idx]
metadata_cols = [df.columns[i] for i in metadata_idx]
weight_cols = [df.columns[i] for i in weight_idx]
binary_cols = [df.columns[i] for i in binary_idx]

embedding_cols = df.drop(remove_cols+metadata_cols+weight_cols+binary_cols, axis=1).columns.to_list()

for col in binary_cols:
    df[col+'_01'] = pd.factorize(df[col])[0]

def normalize_row(row):
    if sum(row) != 1: return row / (np.sum(row) - 1)
    return row 

def generate_binary_weights(col):
    responses = np.array(df[col].to_list())
    
    binary_matrix = (responses[:, None] == responses).astype(int)
    
    return np.apply_along_axis(normalize_row, axis=1, arr=binary_matrix)

def generate_embed_weights(col):
    responses = df[col].to_list()

    OAI_embeddings = [OAI_embedder.embed_query(resp) for resp in responses]
    ST_embeddings = ST_embedder.encode(responses)

    OAI_weights = np.apply_along_axis(normalize_row, axis=1, arr=cosine_similarity(OAI_embeddings))
    ST_weights = np.apply_along_axis(normalize_row, axis=1, arr=cosine_similarity(ST_embeddings))
    return (OAI_weights + ST_weights)/2


col_to_weight = {
    **{col : generate_embed_weights(col) for col in embedding_cols},
    **{col : generate_binary_weights(col+'_01') for col in binary_cols}
}

def mean_edges(question_weights):
    weight_matrices = col_to_weight.values()
    mean_weights = list(question_weights.values())

    return sum(
        mean_weights[i] * weight_matrix for i, weight_matrix in enumerate(weight_matrices)
    ) / sum(mean_weights)


question_weights = {
    **{col : weight for col, weight in zip(embedding_cols, [1, 2, 1, 1, 2])},
    **{col : weight for col, weight in zip(binary_cols, [3, 2])}
}

# MANIPULACAO DOS PESOS DE CADA TOPICO

matching_edges = mean_edges(question_weights)

def get_id(email):
    return df.index[df[email_col_name] == email][0]

emails = df[email_col_name].to_list()

n_person = df.shape[0]
n_son = int(0.7*n_person)
n_par = n_person - n_son
remaining = n_son - n_par

son_idx = np.random.choice(np.arange(0, n_person), n_son, replace=False)
parent_idx = np.setdiff1d(np.arange(0, n_person), son_idx)

arr = np.ones(n_par, dtype=int)

while remaining:
    idx = np.random.randint(0, n_par)

    if arr[idx] < max_godchildren:
        arr[idx] += 1
        remaining -= 1

G = nx.Graph()

sons = [f'{emails[i]}' for i in son_idx]
parents = [f'{emails[i]}' for i in parent_idx]

parent_duplicates = [f'{parents[i]}|||{j}' for i in range(len(parents)) for j in range(arr[i])]

G.add_nodes_from(sons, bipartite=0)  
G.add_nodes_from(parent_duplicates, bipartite=1)  


for son in sons:
    for parent in parent_duplicates:
        weight = matching_edges[get_id(son)][get_id(parent.split('|||')[0])]
        G.add_edge(son, parent, weight=weight)

matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
parent_sons = {}
    
for son, parent in matching:
    if("|||" not in parent):
        son, parent = parent, son
    parent_base = parent.split("|||")[0]
    if parent_base not in parent_sons:
        parent_sons[parent_base] = [] 
    parent_sons[parent_base].append(son)

for parent, son in parent_sons.items():
    print(f'pai: {parent}\n    filhos: {' '.join(son)}')
