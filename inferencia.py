import pandas as pd
import numpy as np
from pickle import load

# 1. CARREGAR OS MODELOS E METADADOS SALVOS NO TREINAMENTO
modelo_normalizador = load(open('modelo_normalizador.model', 'rb'))
boston_cluster = load(open('boston_cluster.model', 'rb'))
metadados = load(open('colunas_treinamento.model', 'rb'))

colunas_originais = metadados['colunas_originais']
colunas_num = metadados['colunas_num']
colunas_cat = metadados['colunas_cat']
colunas_cat_dummies = metadados['colunas_cat_dummies']

# 2. RECEBER DADOS DE UMA NOVA RESIDÊNCIA
# Valores na ordem original: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
dados_nova_residencia = [[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98, 24.0]]
nova_residencia_df = pd.DataFrame(dados_nova_residencia, columns=colunas_originais)

print("--- DADOS DA NOVA RESIDÊNCIA ---")
print(nova_residencia_df.to_string(index=False))

# 3. NORMALIZAR OS DADOS DA NOVA RESIDÊNCIA
# 3.1 Separar e normalizar parte quantitativa
nova_num = nova_residencia_df[colunas_num]
nova_num_norm = pd.DataFrame(
    modelo_normalizador.transform(nova_num),
    columns=colunas_num
)

# 3.2 Separar e normalizar parte categórica (garantindo todas as colunas dummy do treino)
nova_cat = nova_residencia_df[colunas_cat].astype(int).astype(str)
nova_cat_norm = pd.get_dummies(nova_cat, prefix=colunas_cat, dtype=int)
nova_cat_norm = nova_cat_norm.reindex(columns=colunas_cat_dummies, fill_value=0)

# 3.3 Juntar numéricos e categóricos normalizados
nova_residencia_normalizada = nova_num_norm.join(nova_cat_norm)

# 4. INFERIR A QUAL GRUPO (CLUSTER) A NOVA RESIDÊNCIA PERTENCE
cluster_inferido = boston_cluster.predict(nova_residencia_normalizada)[0]
print(f"\nCluster (Grupo) inferido para a nova residência: {cluster_inferido}")

# 5. EXIBIR OS DADOS DESNORMALIZADOS DO CENTROIDE INFERIDO
centroide_normalizado = boston_cluster.cluster_centers_[cluster_inferido]
qtd_colunas_num = len(colunas_num)

# 5.1 Separar as partes numérica e categórica do vetor do centroide
centroide_num_norm = [centroide_normalizado[:qtd_colunas_num]]
centroide_cat_norm = pd.DataFrame(
    [centroide_normalizado[qtd_colunas_num:]],
    columns=colunas_cat_dummies
)

# 5.2 Desnormalizar parte quantitativa com inverse_transform
centroide_num_natural = pd.DataFrame(
    modelo_normalizador.inverse_transform(centroide_num_norm),
    columns=colunas_num
)

# 5.3 Desnormalizar parte categórica com pd.from_dummies
# Como o centroide contém médias entre 0 e 1, definimos como 1 a categoria predominante (maior valor)
centroide_cat_binario = centroide_cat_norm.copy()
for prefixo in colunas_cat:
    cols_prefixo = [c for c in centroide_cat_binario.columns if c.startswith(prefixo + '_')]
    coluna_vencedora = centroide_cat_binario[cols_prefixo].idxmax(axis=1).values[0]
    for c in cols_prefixo:
        centroide_cat_binario[c] = 1 if c == coluna_vencedora else 0

centroide_cat_natural = pd.from_dummies(centroide_cat_binario, sep='_')

# 5.4 Juntar tudo e exibir na ordem original das colunas da base Boston Housing
centroide_desnormalizado = centroide_num_natural.join(centroide_cat_natural)
centroide_desnormalizado = centroide_desnormalizado[colunas_originais]

print(f"\n--- DADOS DESNORMALIZADOS DO CENTROIDE (GRUPO {cluster_inferido}) ---")
print(centroide_desnormalizado.round(4).to_string(index=False))