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

# 2) Explorar os dados (compreender)
print(df.head())
print(df.info())

# 3) Pré-processamento
# Converter a coluna 'shuffle' de True/False para 1/0
if df['shuffle'].dtype == 'bool':
    df['shuffle'] = df['shuffle'].astype(int)
else:
    df['shuffle'] = df['shuffle'].map({True:1, False:0})

# Selecionar apenas algumas features para regressão
selected_features = ['shuffle', 'skipped', 'reason_start', 'reason_end']

# Codificar variáveis categóricas
for col in ['reason_start', 'reason_end']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df[selected_features]
y = df['ms_played']

# 4) Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5) Aplicar três algoritmos de regressão
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso()
}

# Guardar resultados de regressão para o PDF
regression_results = []

print("\nResultados da Regressão:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    regression_results.append(f"{name}: MSE = {mse:.2f}, R2 = {r2:.4f}")
    print(f"{name}: MSE = {mse:.2f}, R2 = {r2:.4f}")

# 6) Clusterização com KMeans e DBSCAN

# Variáveis para clusterização
cluster_features = ['ms_played', 'shuffle']
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[cluster_features])

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_cluster)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_cluster)

# Guardar resultados de clusterização para o PDF
kmeans_n_clusters = len(np.unique(kmeans_labels))
dbscan_n_clusters = len(np.unique(dbscan_labels))

print("\nQuantidade de clusters encontrados pelo KMeans:", kmeans_n_clusters)
print("Quantidade de clusters encontrados pelo DBSCAN:", dbscan_n_clusters)

# --- Gerar o PDF ---
with PdfPages('relatorio_spotify.pdf') as pdf:
    # Gráficos de clusterização
    plt.figure(figsize=(14,6))
    
    # KMeans
    plt.subplot(1,2,1)
    plt.scatter(X_cluster[:,0], X_cluster[:,1], c=kmeans_labels, cmap='viridis')
    plt.title('KMeans Clustering')
    plt.xlabel('ms_played (padronizado)')
    plt.ylabel('shuffle (padronizado)')
    
    # DBSCAN
    plt.subplot(1,2,2)
    plt.scatter(X_cluster[:,0], X_cluster[:,1], c=dbscan_labels, cmap='plasma')
    plt.title('DBSCAN Clustering')
    plt.xlabel('ms_played (padronizado)')
    plt.ylabel('shuffle (padronizado)')
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Página de texto com resultados
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    texto_resultados = "Resultados da Regressão:\n\n"
    for result in regression_results:
        texto_resultados += result + "\n"

    texto_resultados += "\nResultados da Clusterização:\n\n"
    texto_resultados += f"Quantidade de clusters encontrados pelo KMeans: {kmeans_n_clusters}\n"
    texto_resultados += f"Quantidade de clusters encontrados pelo DBSCAN: {dbscan_n_clusters}\n"

    ax.text(0.05, 0.95, texto_resultados, verticalalignment='top', fontsize=12)
    pdf.savefig()
    plt.close()

print("\nRelatório gerado: relatorio_spotify.pdf")
