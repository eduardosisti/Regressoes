import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('admissao.csv')
# print(df.head()) # Mostra as 5 primeiras linhas

positive = df[df['Admitido'].isin([1])]
negative = df[df['Admitido'].isin([0])]

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(positive['Exame1'], positive['Exame2'], s=50, c='g', marker='o', label='Admitido')
ax.scatter(negative['Exame1'], negative['Exame2'], s=50, c='r', marker='x', label='Não Admitido')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show() # Mostra o gráfico PyCharm

# SEPARANDO X de y
n_features = len(df.columns)-1

X = np.array(df.drop('Admitido',1))
y = df.iloc[:,n_features:n_features+1].values

# Guardando os vetores de média e desvio padrão na padronização para classificação futura
mean = X.mean(axis=0)
std = X.std(axis=0)

# Normalização de dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# CRIANDO X-zero com valores = 1
def insert_ones(X):
    ones = np.ones([X.shape[0],1])
    return np.concatenate((ones,X),axis=1)

# Criando um vetor de W's baseado na quantidades de features
w = np.random.rand(1,n_features+1) ## valores entre 0 e 1

# SIGMOID
def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

# BINARY CROSS ENTROPY
def binary_cross_entropy(w, X, y):
    m = len(X)

    parte1 = np.multiply(-y, np.log(sigmoid(X @ w.T)))
    parte2 = np.multiply((1 - y), np.log(1 - sigmoid(X @ w.T)))

    somatorio = np.sum(parte1 - parte2)

    return somatorio / m

# GRADIENT DESCENDENT
def gradient_descent(w, X, y, alpha, epoch):
    cost = np.zeros(epoch)
    for i in range(epoch):
        w = w - (alpha / len(X)) * np.sum((sigmoid(X @ w.T) - y) * X, axis=0)
        cost[i] = binary_cross_entropy(w, X, y)

    return w, cost

# INICIALIZANDO

X = insert_ones(X)

alpha= 0.01 # taxa de aprendizado
epoch = 10000
w, cost = gradient_descent(w, X, y, alpha, epoch)

# PLOTANDO O CUSTO
fig, ax = plt.subplots()
ax.plot(np.arange(epoch), cost, 'r')
ax.set_xlabel('Iterações')
ax.set_ylabel('Custo')
ax.set_title('Erro vs. Epoch')


#VALORES FINAIS DE W
print(w)

def predict(w, X, threshold=0.5):
    p = sigmoid(X @ w.T) >= threshold
    return (p.astype('int'))

# REALIZANDO PREDIÇÕES
# Exame 1 = 20
# Exame 2 = 85
estudante1 = np.array([[20,85]])
estudante1 = (estudante1 - mean)/std
estudante1 = insert_ones(estudante1)

# PROBABILIDADE
print(sigmoid(estudante1@ w.T))

# PREDIÇÃO
print(predict(w, estudante1))
