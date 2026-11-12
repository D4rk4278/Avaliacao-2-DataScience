# Avaliação 2º Bimestre - Data Science (K-Means Boston Housing)

**Aluno:** Matheus Eduardo Ducati  
**RA:** 47614137  
**Disciplina:** Data Science  

## Estrutura do Repositório
* `treinamento.py`: Limpeza de dados ausentes, normalização quantitativa (`MinMaxScaler`) e categórica (`get_dummies`), definição do número ótimo de grupos pelo Método do Cotovelo (distâncias) e treinamento do modelo `KMeans`.
* `inferencia.py`: Normalização dos dados de uma nova residência, predição do cluster correspondente e exibição do centroide desnormalizado.
* `HousingData.csv`: Base de dados original utilizada no treinamento.
* `*.model`: Modelos e metadados persistidos com `pickle`.
