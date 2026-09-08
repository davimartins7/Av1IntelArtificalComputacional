import numpy as np
import matplotlib.pyplot as plt

#ler os dados do arquivo
dados = np.genfromtxt("arsenio_dataset (1).csv", delimiter=",", names=True)

idade = dados["Idade"]
sexo = dados["Sexo"]
beber = dados["Uso_Beber"]
cozinhar = dados["Uso_Cozinhar"]
arsenio_agua = dados["Arsenio_Agua"]
arsenio_unhas = dados["Arsenio_Unhas"]

print("Quantidade de dados:", idade.shape[0])


#regressao linear multipla
class MRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.beta = None
        self.N = X.shape[0]

    def fit(self):
        self.X = np.column_stack((np.ones(self.N), self.X))

        self.beta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y

        return self

    def predict(self, X_new):
        N = X_new.shape[0]

        X_new = np.column_stack((np.ones(N), X_new))

        return X_new @ self.beta


#variaveis usadas no modelo
X = np.column_stack((idade, beber, cozinhar, arsenio_agua))

modelo = MRegression(X, arsenio_unhas)

modelo.fit()

print("\nValores dos parametros:")
print(modelo.beta)


#previsoes para todos os dados
y_pred = modelo.predict(X)

print("\nValores previstos:")
print(y_pred)


#funcao R2
def r2_score(y_true, y_pred):
    numerador = np.sum((y_true - y_pred)**2)
    denominador = np.sum((y_true - np.mean(y_true))**2)

    r_score = 1 - (numerador / denominador)

    return r_score


#R2 ajustado
def r2_ajustado(r2, n, p):
    return 1 - ((1-r2) * (n-1) / (n-p-1))


#MSE
def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)


#RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


#MAE
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))


r2 = r2_score(arsenio_unhas, y_pred)

#4 variaveis independentes
r2_adj = r2_ajustado(r2, len(arsenio_unhas), 4)

print("\nMetricas do modelo completo:")
print("R2:", r2)
print("R2 ajustado:", r2_adj)
print("MSE:", mse(arsenio_unhas, y_pred))
print("RMSE:", rmse(arsenio_unhas, y_pred))
print("MAE:", mae(arsenio_unhas, y_pred))


#previsao solicitada no problema
novo_dado = np.array([[30, 5, 5, 0.135]])

previsao = modelo.predict(novo_dado)

print("\nPrevisao para o novo caso:")
print(previsao)



X_alternativo = np.column_stack((arsenio_agua))

modelo_alternativo = MRegression(X_alternativo.reshape(-1, 1), arsenio_unhas)

modelo_alternativo.fit()

y_pred_alternativo = modelo_alternativo.predict(
    X_alternativo.reshape(-1, 1)
)

print("\nParametros do modelo alternativo:")
print(modelo_alternativo.beta)

r2_alternativo = r2_score(arsenio_unhas, y_pred_alternativo)

#somente uma variavel independente
r2_adj_alternativo = r2_ajustado(
    r2_alternativo,
    len(arsenio_unhas),
    1
)

print("\nMetricas do modelo alternativo:")
print("R2:", r2_alternativo)
print("R2 ajustado:", r2_adj_alternativo)
print("MSE:", mse(arsenio_unhas, y_pred_alternativo))
print("RMSE:", rmse(arsenio_unhas, y_pred_alternativo))
print("MAE:", mae(arsenio_unhas, y_pred_alternativo))



residuos = arsenio_unhas - y_pred

print("\nTabela de residuos:")
print("Observacao | Valor observado | Valor ajustado | Residuo")

for i in range(len(arsenio_unhas)):
    print(
        i+1,
        "|",
        round(arsenio_unhas[i], 6),
        "|",
        round(y_pred[i], 6),
        "|",
        round(residuos[i], 6)
    )



#nesse caso nao colocamos a coluna de 1
X_zero = np.column_stack(
    (idade, beber, cozinhar, arsenio_agua)
)

beta_zero = np.linalg.pinv(
    X_zero.T @ X_zero
) @ X_zero.T @ arsenio_unhas

y_pred_zero = X_zero @ beta_zero

r2_zero = r2_score(arsenio_unhas, y_pred_zero)
rmse_zero = rmse(arsenio_unhas, y_pred_zero)

print("\nModelo com intercepto igual a zero:")
print("Parametros:")
print(beta_zero)

print("R2:", r2_zero)
print("RMSE:", rmse_zero)



plt.scatter(y_pred, residuos)

plt.axhline(0)

plt.xlabel("Valores ajustados")
plt.ylabel("Residuos")

plt.title("Residuos x Valores ajustados")

plt.show()