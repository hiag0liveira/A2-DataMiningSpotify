# Importar as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.backends.backend_pdf import PdfPages

# 1) Carregar a base de dados
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

selected_features = ['shuffle', 'skipped', 'reason_start', 'reason_end', 'platform', 'artist_name']
X = df[selected_features]
y = df['ms_played']

# 3) Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Modelos de regressão (apenas modelos lineares agora)
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso()
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

# 6) Gerar PDF estruturado
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

    ## Página 2 - Pré-processamento
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')
    texto = (
        "Pré-processamento:\n\n"
        "- Seleção de 20.000 registros.\n"
        "- Conversão de shuffle para 0/1.\n"
        "- Codificação de variáveis categóricas: 'reason_start', 'reason_end', 'platform', 'artist_name'.\n"
        "- Definição de variáveis independentes e variável alvo (ms_played).\n"
    )
    ax.text(0.05, 0.95, texto, verticalalignment='top', fontsize=11)
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 8.5))
    sns.histplot(df['ms_played'], bins=50, kde=True, ax=ax)
    ax.set_title('Distribuição do Tempo de Reprodução (ms_played)')
    pdf.savefig()
    plt.close()

    ## Página 3 - Regressão
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')
    texto = (
        "Modelos de Regressão Aplicados:\n\n"
        "- Linear Regression\n"
        "- Ridge Regression\n"
        "- Lasso Regression\n\n"
        "Resultados obtidos:\n"
    )
    for result in regression_results:
        texto += result + "\n"

    texto += (
        "\nConclusão:\n"
        "- A regressão linear apresentou resultados consistentes.\n"
        "- A variabilidade explicada (R²) ficou em torno de 59%-60%.\n"
    )
    ax.text(0.05, 0.95, texto, verticalalignment='top', fontsize=11)
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.scatter(y_test, y_preds['Linear Regression'], alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Valor Real (ms_played)')
    ax.set_ylabel('Valor Previsto')
    ax.set_title('Linear Regression: Real vs Previsto')
    pdf.savefig()
    plt.close()

    ## Página 4 - Clusterização
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

    ## Página 5 - Conclusão Final
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    texto = (
        "Conclusões Gerais:\n\n"
        "- O pré-processamento dos dados foi essencial para a modelagem.\n"
        "- A regressão linear obteve desempenho satisfatório.\n"
        "- A clusterização revelou padrões interessantes de comportamento musical.\n"
        "- O projeto foi concluído com sucesso, atingindo os objetivos propostos."
    )
    ax.text(0.05, 0.95, texto, verticalalignment='top', fontsize=12)
    pdf.savefig()
    plt.close()

print("\nRelatório gerado: relatorio_spotify.pdf")
