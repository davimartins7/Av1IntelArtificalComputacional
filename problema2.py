import numpy as np
import matplotlib.pyplot as plt

#ler dados
dados = np.genfromtxt("dose_radiacao_expandido.csv", delimiter=",", skip_header=1)

dose = dados[:,1]
corrente = dados[:,2]
tempo = dados[:,3]

print("Quantidade de dados:", dose.shape[0])


#regressao multipla
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


X = np.column_stack((corrente, tempo))

modelo = MRegression(X, dose)
modelo.fit()

print("\nValores Parametros")
print(modelo.beta)

y_pred = modelo.predict(X)

print("\nValores previstos")
print(y_pred)


#r2
def r2_score(y_true, y_pred):
    numerador = np.sum((y_true - y_pred)**2)
    denominador = np.sum((y_true - np.mean(y_true))**2)
    r_score = 1 - (numerador / denominador)
    return r_score


#r2 ajustado
def r2_ajustado(r2, n, p):
    return 1 - ((1-r2) * (n-1) / (n-p-1))


#mse
def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)


#rmse
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


#mae
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))


r2 = r2_score(dose, y_pred)
r2_adj = r2_ajustado(r2, len(dose), 2)

print("\nMetricas modelo completo")
print("R2:", r2)
print("R2 ajustado:", r2_adj)
print("MSE:", mse(dose, y_pred))
print("RMSE:", rmse(dose, y_pred))
print("MAE:", mae(dose, y_pred))


#previsao
novo_dado = np.array([[15, 5]])
previsao = modelo.predict(novo_dado)

print("\nPrevisao")
print(previsao)


#modelo so com corrente
X2 = corrente.reshape(-1, 1)

modelo2 = MRegression(X2, dose)
modelo2.fit()

y_pred2 = modelo2.predict(X2)

print("\nParametros modelo so corrente")
print(modelo2.beta)

r2_2 = r2_score(dose, y_pred2)
r2_adj_2 = r2_ajustado(r2_2, len(dose), 1)

print("\nMetricas modelo so corrente")
print("R2:", r2_2)
print("R2 ajustado:", r2_adj_2)
print("MSE:", mse(dose, y_pred2))
print("RMSE:", rmse(dose, y_pred2))
print("MAE:", mae(dose, y_pred2))


#residuos
residuos = dose - y_pred

print("\nTabela de residuos")
print("Observacao | Valor observado | Valor ajustado | Residuo")

for i in range(len(dose)):
    print(i+1, "|", round(dose[i], 6), "|", round(y_pred[i], 6), "|", round(residuos[i], 6))


#intercepto = 0
X_zero = np.column_stack((corrente, tempo))

beta_zero = np.linalg.pinv(X_zero.T @ X_zero) @ X_zero.T @ dose

y_pred_zero = X_zero @ beta_zero

r2_zero = r2_score(dose, y_pred_zero)
rmse_zero = rmse(dose, y_pred_zero)

print("\nModelo com intercepto igual a zero")
print("Parametros")
print(beta_zero)
print("R2:", r2_zero)
print("RMSE:", rmse_zero)


#graf residuos
plt.scatter(y_pred, residuos)
plt.axhline(0)

plt.xlabel("Valores ajustados")
plt.ylabel("Residuos")
plt.title("Residuos x Valores ajustados")

plt.show()