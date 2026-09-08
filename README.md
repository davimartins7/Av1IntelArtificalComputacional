# Modelos de Regressão Linear: Arsênio e Radiação 

Repositório contendo a implementação matemática de modelos de Regressão Linear Múltipla e Simples. Este projeto foi desenvolvido como requisito avaliativo para a disciplina de Inteligência Artificial Computacional da Universidade de Fortaleza (UNIFOR).

## 🎯 Objetivo do Projeto
Aplicar os conceitos de regressão linear para resolver dois problemas preditivos utilizando álgebra linear e calcular as métricas de avaliação manualmente. Em estrita conformidade com as regras do projeto, **não foram utilizadas** bibliotecas com implementações prontas como `scikit-learn` ou `pandas`[cite: 1]. Todo o cálculo matricial foi construído do zero.

## 🗂️ Estrutura do Repositório
* `problema1.py`: Modelo preditivo para avaliar concentrações de arsênio nas unhas como indicador de ingestão de água contaminada. Realiza o treinamento usando variáveis como idade, uso da água para beber/cozinhar e arsênio na água[cite: 1].
* `problema2.py`: Estudo do efeito da inspeção de raios X em circuitos integrados, prevendo a dose de radiação em função da corrente e do tempo de exposição[cite: 1].
* `arsenio_dataset (1) (1).csv`: Base de dados contendo as 21 amostras do estudo piloto sobre concentração de arsênio[cite: 1].
* `dose_radiacao_expandido (1).csv`: Base de dados contendo os registros de corrente, tempo e dose de radiação[cite: 1].

## 🛠️ Tecnologias Utilizadas
* **Python**
* **NumPy:** Manipulação de matrizes e resolução da Equação Normal para encontrar os parâmetros $\beta$.
* **Matplotlib:** Geração dos gráficos de dispersão (Análise de Resíduos).

## 🚀 Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/davimartins7/Av1IntelArtificalComputacional.git


2. Instale as dependências básicas:
```bash
pip install numpy matplotlib
```

3. Execute os scripts diretamente no terminal:
```bash
python problema1.py
python problema2.py
```


## 📊 Métricas Implementadas

Os algoritmos realizam previsões para novos cenários e imprimem no terminal as seguintes métricas matemáticas construídas manualmente:

- **Coeficiente de Determinação ($R^2$)**
- **$R^2$ Ajustado**
- **Erro Quadrático Médio (MSE)**
- **Raiz do Erro Quadrático Médio (RMSE)**
- **Erro Absoluto Médio (MAE)**
- **Tabela formatada de Resíduos** ($y_i$, $\hat{y}_i$, $e_i$)

---

**Autor:** Davi Martins
