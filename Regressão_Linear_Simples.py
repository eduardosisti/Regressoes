import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

# Importar o arquvivo CSV
df1 = pd.read_csv('plano_saude.csv')

# Armazenar as variaveis dos modelo em X e y
X = df1.iloc[:, 0].values
y = df1.iloc[:, 1].values

print(X, y)

# Correlação
correlacao = np.corrcoef(X, y) # Ou df1.corr()

# Converte os dados em array
X = X.reshape(-1, 1)

# Ajuste
regressor = LinearRegression()
regressor.fit(X, y)

# y = B0 + B1*X
# Coeficientes
# B0
regressor.intercept_

# B1
regressor.coef_

# Plotar em um gráfico
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color="green")
plt.title("Regressão Linear Simples")
plt.xlabel("idade")
plt.ylabel("Custo")
plt.show()

# Utilizando o modelo
predict1 = regressor.intercept_ + regressor.coef_*40   # 40 = variavel de escolha[idade]
print(predict1)

# Mostra o R²
# Coeficiente R²: diz o quanto o meu modelo explica seus resultados. É um valor entre 0 e 1. Quanto mais próximo de 1, melhor. (Nem sempre este racíocinio e válido, necessita de análise)
score = regressor.score(X, y)
print(score)

# Distância dos dados reais até a reta preditiva, com este método é possível ver o R-Square(R²)
vizualizador = ResidualsPlot(regressor)
vizualizador.fit(X, y)
vizualizador.poof()