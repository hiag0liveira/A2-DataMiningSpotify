# Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.backends.backend_pdf import PdfPages

# 1) Carregar dados
df = pd.read_csv('./data/spotify_history.csv')
df = df.sample(n=20000, random_state=42)

# 2) Pré-processamento
if df['shuffle'].dtype == 'bool':
    df['shuffle'] = df['shuffle'].astype(int)
else:
    df['shuffle'] = df['shuffle'].map({True:1, False:0})

for col in ['reason_start', 'reason_end', 'platform', 'artist_name']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Verificar dados nulos
missing_values = df.isnull().sum()
total_missing = missing_values.sum()

# Se houver linhas com dados nulos, remove
if total_missing > 0:
    df = df.dropna()

selected_features = ['shuffle', 'skipped', 'reason_start', 'reason_end', 'platform', 'artist_name']
X = df[selected_features]
y = df['ms_played']

# 3) Dividir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Modelos de regressão
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet(),
    'Bayesian Ridge Regression': BayesianRidge()
}

regression_results = []
y_preds = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    regression_results.append(f"{name}: MSE = {mse:.2f}, R2 = {r2:.4f}")
    y_preds[name] = y_pred

# 5) Clusterização
cluster_features = ['ms_played', 'shuffle']
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_cluster)

dbscan = DBSCAN(eps=0.7, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_cluster)

kmeans_n_clusters = len(np.unique(kmeans_labels))
dbscan_n_clusters = len(np.unique(dbscan_labels))

# 6) Gerar PDF
with PdfPages('relatorio_spotify.pdf') as pdf:

    ## Página 1 - Introdução
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    texto = (
        "Projeto: Mineração de Dados - Spotify Listening History\n\n"
        "Objetivo:\n"
        "- Prever o tempo de reprodução (ms_played) a partir do comportamento de usuário.\n"
        "- Identificar padrões de reprodução musical usando técnicas de clusterização.\n"
    )
    ax.text(0.05, 0.95, texto, verticalalignment='top', fontsize=13)
    pdf.savefig()
    plt.close()

    ## Página 2 - Pré-processamento (Texto detalhado)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    texto = (
        "Pré-processamento dos Dados:\n\n"
        "- Seleção de 20.000 registros aleatórios.\n"
        "- Conversão da variável 'shuffle' para valores 0 (False) e 1 (True).\n"
        "- Codificação numérica de variáveis categóricas ('reason_start', 'reason_end', 'platform', 'artist_name').\n"
        "- Verificação de dados faltantes:\n"
        f"  - Total de valores nulos encontrados: {total_missing}\n"
    )
    if total_missing > 0:
        texto += "- Linhas com valores nulos foram removidas.\n"
    else:
        texto += "- Nenhuma linha precisou ser removida.\n"

    texto += "- Dados prontos para a divisão entre treino e teste."

    ax.text(0.05, 0.95, texto, verticalalignment='top', fontsize=11)
    pdf.savefig()
    plt.close()

    ## Página 3 - Distribuição ms_played
    fig, ax = plt.subplots(figsize=(11, 8.5))
    sns.histplot(df['ms_played'], bins=50, kde=True, ax=ax)
    ax.set_title('Distribuição do Tempo de Reprodução (ms_played)')
    pdf.savefig()
    plt.close()

    ## Página 4 - Regressão (descrição e resultados)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    texto = (
        "Modelos de Regressão Aplicados:\n\n"
        "- Linear Regression: método clássico para prever uma variável contínua.\n"
        "- Ridge Regression: regressão linear com regularização para evitar overfitting.\n"
        "- Lasso Regression: regressão linear que pode zerar coeficientes irrelevantes.\n"
        "- ElasticNet: combinação de Ridge e Lasso.\n"
        "- Bayesian Ridge: incorpora probabilidade na previsão dos coeficientes.\n\n"
        "Resultados obtidos:\n"
    )
    for result in regression_results:
        texto += result + "\n"

    texto += "\nConclusão:\n- Modelos lineares e regularizados apresentaram desempenhos muito próximos."

    ax.text(0.05, 0.95, texto, verticalalignment='top', fontsize=10)
    pdf.savefig()
    plt.close()

    ## Página 5 - Gráficos de Regressão Real vs Previsto
    for name, preds in y_preds.items():
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.scatter(y_test, preds, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Valor Real (ms_played)')
        ax.set_ylabel('Valor Previsto')
        ax.set_title(f'{name}: Real vs Previsto')
        pdf.savefig()
        plt.close()

    ## Página 6 - Clusterização
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')
    texto = (
        "Clusterização Aplicada:\n\n"
        "- KMeans (n_clusters=3)\n"
        "- DBSCAN (eps=0.7, min_samples=5)\n\n"
        f"Clusters encontrados:\n"
        f"- KMeans: {kmeans_n_clusters} clusters\n"
        f"- DBSCAN: {dbscan_n_clusters} clusters\n"
    )
    ax.text(0.05, 0.95, texto, verticalalignment='top', fontsize=11)
    pdf.savefig()
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(11, 8.5))
    
    axs[0].scatter(X_cluster[:,0], X_cluster[:,1], c=kmeans_labels, cmap='viridis')
    axs[0].set_title('KMeans Clustering')
    axs[0].set_xlabel('ms_played (padronizado)')
    axs[0].set_ylabel('shuffle (padronizado)')

    axs[1].scatter(X_cluster[:,0], X_cluster[:,1], c=dbscan_labels, cmap='plasma')
    axs[1].set_title('DBSCAN Clustering')
    axs[1].set_xlabel('ms_played (padronizado)')
    axs[1].set_ylabel('shuffle (padronizado)')

    plt.tight_layout()
    pdf.savefig()
    plt.close()
    ## Nova página - Comparação de desempenho dos modelos (barras % de R²)
    fig, ax = plt.subplots(figsize=(11, 8.5))

    # Preparar dados
    model_names = []
    r2_scores = []
    for result in regression_results:
        parts = result.split(", R2 = ")
        model_name = parts[0].split(":")[0]
        r2_value = float(parts[1])
        model_names.append(model_name)
        r2_scores.append(r2_value * 100)  # Convertendo para %

    ax.barh(model_names, r2_scores, color='skyblue')
    ax.set_xlim(0, 100)
    ax.set_xlabel('R² (%)')
    ax.set_title('Comparação de Desempenho dos Modelos de Regressão')
    for index, value in enumerate(r2_scores):
        ax.text(value + 1, index, f'{value:.2f}%', va='center')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    ## Página 7 - Conclusão Final
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    texto = (
        "Conclusões Gerais:\n\n"
        "- O pré-processamento dos dados garantiu qualidade na modelagem.\n"
        "- Modelos lineares e regularizados tiveram desempenhos semelhantes.\n"
        "- A clusterização identificou padrões de comportamento musical.\n"
        "- O projeto foi concluído com sucesso."
    )
    ax.text(0.05, 0.95, texto, verticalalignment='top', fontsize=12)
    pdf.savefig()
    plt.close()

print("\nRelatório gerado: relatorio_spotify.pdf")
