import uuid

import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


setup = {
    'drop' : [0,1],
    'personal' : [0,1],
    'veterano' : [3],
}

topic_to_idx = {
    'multiple_choice' : [2],
    'binary' : [3,4],
    'numerical' : [6],
    'embedding' : [5,7,8,9],
}


total_df = pd.read_csv('../test.csv')

total_df.columns = total_df.columns.str.replace(r'[^\w]', '_', regex=True)

total_df['ID'] = [str(uuid.uuid4()) for _ in range(len(total_df))]
total_df = total_df.drop(total_df.columns[setup['drop']], axis=1)

personal_df = total_df.iloc[:, setup['personal']+setup['veterano']].copy()

personal_df['ID'] = total_df['ID']

df = total_df.drop(total_df.columns[setup['personal']], axis=1).sample(frac=1).reset_index(drop=True)


total_df


df.head()


personal_df.head()


padrinho_col = df.columns[1]
curso_col = df.columns[0]
nome_col = personal_df.columns[1]


topic_to_cols = {
    topic : [df.columns[i] for i in idx] 
    for topic, idx in topic_to_idx.items()
}


topic_to_cols   


topic_to_weights = {
    'multiple_choice' : [8],
    'binary' : [2,1],
    'numerical' : [1],
    'embedding' : [1,1,1,1],
}


par_idx = df[df[padrinho_col] == 'Veterane'].index.to_list()
chi_idx = df.index.difference(par_idx).to_list()


def apply_restrictions():
    pass


def normalize_func(row):
    # exp_values = np.exp(row - np.max(row))  
    # return exp_values / np.sum(exp_values)

    return row / np.sum(row)
    # mais contrastanete pra valores binarios, melhora na aplicacao do peso


def normalize_rows(mat):
    normalized_mat = np.apply_along_axis(
        normalize_func, axis=1, arr=mat
    )

    return normalized_mat


def handle_topic_weights(topic, weight_func):
    col_to_weights = {}
    for col in topic_to_cols[topic]:
        
        par_resp = df.loc[par_idx, col].to_numpy()
        chi_resp = df.loc[chi_idx, col].to_numpy()

        weight_mat = weight_func(chi_resp, par_resp)

        apply_restrictions()

        col_to_weights[col] = normalize_rows(weight_mat)

    return col_to_weights



embedder = SentenceTransformer('all-mpnet-base-v2')


def get_embed_weight(chi_resp, par_resp):
    par_embed = embedder.encode(par_resp)   
    chi_embed = embedder.encode(chi_resp)

    weight_mat = cosine_similarity(chi_embed, par_embed)

    return weight_mat


def num_func(x1, x2):
    return 1 - (x1 - x2)**2


def get_num_weight(chi_resp, par_resp):
    par_numeric = np.array([int(x) if str(x).isnumeric() else 18 for x in par_resp])
    chi_numeric = np.array([int(x) if str(x).isnumeric() else 18 for x in chi_resp])

    combined = np.concatenate((par_numeric, chi_numeric))
    diff = combined.max() - combined.min()
    
    par_numeric = par_numeric / diff
    chi_numeric = chi_numeric / diff

    weight_mat = np.zeros((len(chi_numeric), len(par_numeric)))

    for i, x1 in enumerate(chi_numeric):
        weight_mat[i, :] = np.vectorize(num_func)(x1, par_numeric)

    return weight_mat



def get_binary_weight(chi_resp, par_resp):

    weight_mat = np.zeros((len(chi_resp), len(par_resp)))

    for i, chi in enumerate(chi_resp):
        for j, par in enumerate(par_resp):
            weight_mat[i, j] = chi == par

    return weight_mat


def get_multiple_choice_weight(chi_resp, par_resp):

    weight_mat = np.zeros((len(chi_resp), len(par_resp)))

    par_resp = [x.split(';') for x in par_resp] 
    chi_resp = [x.split(';') for x in chi_resp]
    
    for i, chi in enumerate(chi_resp):
        for j, par in enumerate(par_resp):
            weight_mat[i, j] = any(element in chi for element in par)

    return weight_mat


col_to_weights = {
    **handle_topic_weights('multiple_choice', get_multiple_choice_weight),
    **handle_topic_weights('binary', get_binary_weight),
    **handle_topic_weights('numerical', get_num_weight),
    **handle_topic_weights('embedding', get_embed_weight),
}


col_to_weights


edges = np.zeros((len(chi_idx), len(par_idx)))

for topic, cols in topic_to_cols.items():
    for i, col in enumerate(cols):
        edges += col_to_weights[col] * topic_to_weights[topic][i]

weight_sum = sum(np.sum(weights) for weights in topic_to_weights.values())

edges = edges / weight_sum


edges


par_to_mat = { idx : i for i, idx in enumerate(par_idx) }
chi_to_mat = { idx : i for i, idx in enumerate(chi_idx) }


def distribute_random(n, max_n, max_chi_par):
    distribution = np.ones(n, dtype=int) 
    diff = max_n - n

    available_indices = set(range(n))

    while diff > 0:
        idx = np.random.choice(list(available_indices))
        
        distribution[idx] += 1
        diff -= 1
        
        if distribution[idx] == max_chi_par:
            available_indices.remove(idx)

    return distribution



n_chi = len(chi_idx)
n_par = len(par_idx)

max_chi_par = 2

diff = n_chi - n_par

max_matches = min(n_chi, n_par)*max_chi_par

chi_dist = distribute_random(n_chi, max_matches, max_chi_par)
par_dist = distribute_random(n_par, max_matches, max_chi_par)

chi_ids = np.array([f'{str(chi_idx[i])}_{j}' for i in range(len(chi_idx)) for j in range(chi_dist[i])])
par_ids = np.array([f'{str(par_idx[i])}_{j}' for i in range(len(par_idx)) for j in range(par_dist[i])])
# duplica os pais e filhos pra terem o mesmo numero 

G = nx.Graph()
G.add_nodes_from(chi_ids, bipartite=0)
G.add_nodes_from(par_ids, bipartite=1)

for chi in chi_ids:
    for par in par_ids:
        i = chi_to_mat[int(chi.split('_')[0])]
        j = par_to_mat[int(par.split('_')[0])]

        weight = edges[i, j]
        if int(chi.split('_')[1]) > 0 and int(par.split('_')[0]) > 0:
            weight = 0
        # de for repetido tanto o filho quanto o pai, peso eh zero pra nao repetir 
        # condicao pode ser amenizada pra melhorar o algoritmo em si, mas e ajustavel com os pesos dos topicos

        G.add_edge(chi, par, weight=weight)
        
        # precisa usar id do dataframe, porque matching nao necessariamente sai num formato fixo
        # nao da pra usar relativo à matriz

matching = nx.matching.max_weight_matching(G, maxcardinality=True)

print(matching)




uid_to_row = { row['ID'] : i for i, row in personal_df.iterrows() }
uid_to_row


padrinho_nomes = personal_df[personal_df[padrinho_col] == 'Veterane'][nome_col].to_list()
bixos_nomes = personal_df[personal_df[padrinho_col] == 'Bixe'][nome_col].to_list()

padrinho_to_bixo = { nome : [] for nome in padrinho_nomes }
bixo_to_padrinho = { nome : [] for nome in bixos_nomes }

for p1_idx, p2_idx in matching:
    p1_personal_row = uid_to_row[df.loc[int(p1_idx.split('_')[0]), 'ID']]
    p2_personal_row = uid_to_row[df.loc[int(p2_idx.split('_')[0]), 'ID']]

    p1_nome = personal_df.loc[p1_personal_row, nome_col]
    p2_nome = personal_df.loc[p2_personal_row, nome_col]

    if personal_df.loc[p1_personal_row, padrinho_col] == "Veterane":
        padrinho_to_bixo[p1_nome].append(p2_nome)
        bixo_to_padrinho[p2_nome].append(p1_nome)
    else:
        padrinho_to_bixo[p2_nome].append(p1_nome)
        bixo_to_padrinho[p1_nome].append(p2_nome)


print(padrinho_to_bixo)
print(bixo_to_padrinho)
