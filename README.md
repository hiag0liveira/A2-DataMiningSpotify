# A2 - Mineração de Dados: Spotify Listening History

Repositório contendo o projeto de Mineração de Dados desenvolvido para a disciplina _Mineração de Dados_, cujo objetivo é analisar e prever padrões de comportamento musical a partir do histórico de reprodução do Spotify. O trabalho foi dividido em duas partes:

1. **Implementação e envio do código** (valor 5,0)
2. **Apresentação dos integrantes** (valor 5,0)

---

## Descrição do Projeto

O objetivo principal é realizar uma análise de dados de histórico de músicas, utilizando técnicas de **Machine Learning**. As etapas principais foram:

1. **Coleta** e **exploração** dos dados.
2. **Pré-processamento**, incluindo:
   - Conversão de variáveis booleanas.
   - Codificação de variáveis categóricas em numéricas.
   - Tratamento de valores faltantes.
3. Aplicação de **cinco algoritmos de regressão linear**:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - ElasticNet Regression
   - Bayesian Ridge Regression
4. Avaliação dos modelos com base nas métricas **MSE** (Erro Quadrático Médio) e **R²** (Coeficiente de Determinação).
5. Aplicação de **dois algoritmos de clusterização**:
   - KMeans
   - DBSCAN
6. **Geração automática de um relatório em PDF**, contendo gráficos e análise textual.

---

## Estrutura do Repositório

Dentro do repositório, encontramos a seguinte organização de pastas e arquivos (exemplo):

```
A2-DataMiningSpotify/
├── data/
│   ├── spotify_history.csv
├── venv/                (pasta do ambiente virtual - opcional, não versionado)
├── main.py              (script principal de análise, regressão e clusterização)
├── relatorio_spotify.pdf (relatório em PDF com as análises geradas)
├── requirements.txt     (lista de dependências do projeto)
└── README.md            (este arquivo)
```

- **data/spotify_history.csv**: Base de dados original contendo o histórico de reproduções do Spotify.
- **main.py**: Script Python com todo o fluxo de pré-processamento, regressão, clusterização e geração do relatório.
- **relatorio_spotify.pdf**: Arquivo PDF gerado automaticamente com gráficos e textos analíticos.
- **README.md**: Documento explicativo sobre o projeto (este arquivo).

---

## Como Executar o Projeto

### 1. Clonar o Repositório

No terminal, execute:

```bash
git clone https://github.com/hiag0liveira/A2-DataMiningSpotify.git
cd A2-DataMiningSpotify
```

### 2. Criar e Ativar Ambiente Virtual (Recomendado)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependências

Com o ambiente virtual ativo, rode:

```bash
pip install -r requirements.txt
```

Ou, caso prefira instalar manualmente:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Executar o Script

Para rodar a análise e gerar o relatório:

```bash
python main.py
```

- Ao final da execução, será criado o arquivo `relatorio_spotify.pdf` no diretório do projeto.

---

## Principais Bibliotecas Utilizadas

- **pandas**: Manipulação e análise de dados.
- **numpy**: Operações numéricas e manipulação de arrays.
- **matplotlib** e **seaborn**: Visualização de dados através de gráficos e histogramas.
- **scikit-learn**: Implementação de algoritmos de regressão, clusterização e pré-processamento.

---

## Resultados

Ao final do projeto, são apresentados:

1. **Resultados de regressão** (MSE e R²) para cada modelo testado.
2. **Gráficos de Real vs Previsto** para cada modelo de regressão.
3. **Gráficos de clusterização** (KMeans e DBSCAN).
4. **Comparativo percentual de desempenho** (gráfico de barras com R²).
5. **Conclusões** sobre a qualidade dos modelos e padrões de comportamento detectados.

---

## Créditos / Integrantes do Grupo

- **Hiago De Oliveira**: responsável pelo ambiente de produção e pré processamento.
- **Lucas Rangel**: responsável pela escolha e estudo dos algortimos de clausterização.
- **Matheus Rocha**: responsável pela escolha e estudo dos algortimos de regressão linear.

---
