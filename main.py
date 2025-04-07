import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("datos_clientes.csv")

features = ['Income', 'CreditScore', 'Balance', 'NumProducts', 'Age']
X = df[features]

# Estandarizacion de los datos (PCA es sensible a la escala)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicando PCA y reduciendo a 2 dimensiones para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

# Resultados de la varianza
print("\nResultados de la varianza:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.2%}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7)
plt.title('Visualización de clientes mediante PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)

plt.savefig("results.jpg", dpi=300, bbox_inches='tight')

plt.show()

# Interpretación de los resultados

# Al aplicar PCA a las variables financieras de los clientes (Ingreso, Puntaje de crédito, Saldo, Edad, Número de productos), observamos que las dos primeras componentes principales explican la mayor parte de la variabilidad del conjunto de datos.

# Esto permite reducir las dimensiones de 5 a 2, facilitando la visualización y análisis. En el gráfico resultante, se pueden identificar agrupaciones de clientes con comportamientos financieros similares, lo cual puede ayudar al banco a segmentar clientes, personalizar ofertas o detectar perfiles de riesgo.