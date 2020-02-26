import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Importar o arquvivo CSV
df1 = pd.read_csv('plano_saude.csv')

# Remodelando o vetor
X = df1.iloc[:, 0].values
y = df1.iloc[:, 1].values
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Normalização das variáveis
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Ou

# col = ['idade', 'custo']
# data_scaled = scaler.fit_transform(df1)
# new_data = pd.DataFrame(data_scaled, columns=col)
# print(new_data.head())
# X = new_data.iloc[:, 0].values
# y = new_data.iloc[:, 1].values

#Taxa de aprendizado
alpha = 0.01

w0 = 0.1
w1 = 0.1

def y_hat(x, w0, w1):
    return w0 + w1*x

def plot_line(X, y, w0, w1):
    x_values = [i for i in range(int(min(X)) - 1, int(max(X)) + 2)]
    y_values = [y_hat(x, w0, w1) for x in x_values]
    plt.plot(x_values, y_values, 'r')
    plt.plot(X, y, 'bo')

def MSE(X, y, w0, w1):
    custo = 0
    m = float(len(X))
    for i in range(0, len(X)):
        custo += (y_hat(X[i], w0, w1) - y[i]) ** 2

    return custo / m

def gradient_descent_step(w0, w1, X, y, alpha):
    erro_w0 = 0
    erro_w1 = 0
    m = float(len(X))

    for i in range(0, len(X)):
        erro_w0 += y_hat(X[i], w0, w1) - y[i]
        erro_w1 += (y_hat(X[i], w0, w1) - y[i]) * X[i]

    new_w0 = w0 - alpha * (1 / m) * erro_w0
    new_w1 = w1 - alpha * (1 / m) * erro_w1

    return new_w0, new_w1

interacoes = 400

def gradient_descent(w0, w1, X, y, alpha, interecacoes):
    custo = np.zeros(interacoes)
    for i in range(interacoes):
        w0, w1 = gradient_descent_step(w0, w1, X, y, alpha)
        custo[i] = MSE(X, y, w0, w1)

    return w0, w1, custo

w0, w1, custo = gradient_descent(w0, w1, X, y, alpha, interacoes)

# Plotar taxa de aprendizado do erro quadrático médio por interações (Gradiente descendente)
print(f"w0={w0}, w1={w1}")
fig, ax = plt.subplots()
ax.plot(np.arange(interacoes), custo, 'r')
ax.set_xlabel('Interações')
ax.set_ylabel('Custo')
ax.set_title('MSE vs. Interacoes')
plt.show()

# Plotar reta preditiva
plot_line(X,y, w0, w1)
plt.show()


# Desnormalizar
def previsao(n):
    prev = scaler_y.inverse_transform(y_hat(scaler_X.transform([[n]]), w0, w1))
    return prev

#Teste com idade = 40
print(previsao(40))

