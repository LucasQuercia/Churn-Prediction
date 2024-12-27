import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier  # Alterado para DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Configura a semente aleatória para garantir reprodutibilidade
random.seed(42)

# Carrega os dados
data = pd.read_csv('Churn_Modelling.csv', header=0)
data = data.dropna(axis='rows')  # Remove NAN
classes = np.array(pd.unique(data[data.columns[-1]]), dtype=str)

# Contagem da frequência dos países na coluna 'Geography'
pais_ranking = data['Geography'].value_counts()

# Exibe o ranking de países pela quantidade de ocorrências
print("Ranking de países mais frequentes:")
print(pais_ranking)

# Converte para array NumPy e divide em X (atributos) e y (target)
X = data.drop(['Surname', 'Geography', 'Gender'], axis=1).to_numpy()  # Remove colunas não numéricas
y = data['Exited'].to_numpy()

# Convertendo colunas categóricas 'Geography' e 'Gender' em numéricas usando o LabelEncoder
labelencoder = LabelEncoder()

# Convertendo 'Geography' e 'Gender' para valores numéricos
data['Geography'] = labelencoder.fit_transform(data['Geography'])
data['Gender'] = labelencoder.fit_transform(data['Gender'])

# Exclui as colunas 'Surname', 'Geography' e 'Gender' da variável X
X = data.drop(['Surname', 'Exited'], axis=1).to_numpy()  # Retira 'Surname' e 'Exited' que são irrelevantes

# Normaliza os dados apenas nas colunas numéricas
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divide em conjuntos de treino e teste
p = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=p, random_state=42)

# Inicializa o modelo Árvore de Decisão
decision_tree = DecisionTreeClassifier(random_state=42)  # Pode ajustar parâmetros como max_depth se necessário

# Treina o modelo
decision_tree.fit(X_train, y_train)

# Faz as previsões no conjunto de teste
y_pred = decision_tree.predict(X_test)

# Calcula a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)

# Exibe a acurácia
print(f"Acurácia do modelo Árvore de Decisão: {accuracy * 100:.2f}%")
