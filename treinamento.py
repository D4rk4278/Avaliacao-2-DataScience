import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import math
from pickle import dump

# 1. CARREGAR E LIMPAR A BASE DE DADOS
dados = pd.read_csv('HousingData.csv', sep=',')
# Remover registros com valores ausentes (NaN) para não quebrar o KMeans
dados = dados.dropna().reset_index(drop=True)

# 2. SEPARAR DADOS QUANTITATIVOS E CATEGÓRICOS
# Na base Boston Housing, CHAS (margem do rio: 0 ou 1) e RAD (índice de rodovias) são categóricos.
# (Caso queira considerar apenas CHAS, basta alterar para: colunas_categoricas = ['CHAS'])
colunas_categoricas = ['CHAS', 'RAD']

dados_num = dados.drop(columns=colunas_categoricas)
# Converter categóricas para inteiro e depois texto para o get_dummies reconhecer corretamente
dados_cat = dados[colunas_categoricas].astype(int).astype(str)

# 3. NORMALIZAR DADOS QUANTITATIVOS (MinMaxScaler)
normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_num)
dump(modelo_normalizador, open('modelo_normalizador.model', 'wb'))

dados_num_normalizados = modelo_normalizador.fit_transform(dados_num)
dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=dados_num.columns)

# 4. NORMALIZAR DADOS CATEGÓRICOS (get_dummies) E JUNTAR (join)
dados_cat_normalizados = pd.get_dummies(
    dados_cat,
    prefix=colunas_categoricas,
    dummy_na=False,
    dtype=int
)

# Recompor o DataFrame unindo dados numéricos e categóricos normalizados
dados_normalizados = dados_num_normalizados.join(dados_cat_normalizados)

# Salvar metadados das colunas para garantir a mesma estrutura na inferência
metadados_colunas = {
    'colunas_originais': dados.columns.tolist(),
    'colunas_num': dados_num.columns.tolist(),
    'colunas_cat': colunas_categoricas,
    'colunas_cat_dummies': dados_cat_normalizados.columns.tolist()
}
dump(metadados_colunas, open('colunas_treinamento.model', 'wb'))

# 5. DETERMINAR O NÚMERO ÓTIMO DE GRUPOS (MÉTODO DO COTOVELO - DISTÂNCIAS)
distorcoes = []
K = range(1, 100)
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_normalizados)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_normalizados, cluster_model.cluster_centers_, 'euclidean'),
                axis=1
            ) / dados_normalizados.shape[0]
        )
    )

x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0)**2 + (xn - x0)**2)
    distancias.append(numerador / denominador)

n_clusters_otimo = K[distancias.index(max(distancias))]
print(f"Número ótimo de clusters (grupos) pelo Método do Cotovelo: {n_clusters_otimo}")

# 6. TREINAR E SALVAR O MODELO DEFINITIVO
boston_cluster = KMeans(n_clusters=n_clusters_otimo, random_state=42, n_init=10).fit(dados_normalizados)
dump(boston_cluster, open('boston_cluster.model', 'wb'))
print("Modelo treinado e salvo com sucesso como 'boston_cluster.model'!")