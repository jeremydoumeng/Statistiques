# Implémentations Manuelles d'Algorithmes Statistiques et de Machine Learning

Ce projet contient des implémentations "from scratch" de plusieurs algorithmes fondamentaux de machine learning et de statistiques en Python. Il est conçu à des fins pédagogiques pour comprendre les concepts sous-jacents de ces algorithmes.

## Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Installation](#installation)
3. [Structure du Projet](#structure-du-projet)
4. [Algorithmes Implémentés](#algorithmes-implémentés)
5. [Utilisation](#utilisation)
6. [Dépendances](#dépendances)
7. [Exemples](#exemples)

## Vue d'ensemble

Le projet implémente manuellement plusieurs algorithmes statistiques et de machine learning sans utiliser de bibliothèques spécialisées (comme scikit-learn). Chaque implémentation est accompagnée de commentaires détaillés expliquant la théorie mathématique sous-jacente.

## Installation

```bash
cd statistiques
pip install numpy scipy
```

## Structure du Projet

- `main.py`: Fichier principal contenant toutes les implémentations
- `README.md`: Documentation du projet

## Algorithmes Implémentés

### 1. Fonctions Utilitaires
- `multivariate_normal_pdf`: Calcul de densité de probabilité normale multivariée
- `univariate_normal_pdf`: Calcul de densité de probabilité normale univariée

### 2. Classification (Chapitre 7)
- Classifieur de Bayes théorique
- Plug-in classifieur (QDA)
- Analyse Discriminante Linéaire (LDA)
- Bayes Naïf Gaussien

### 3. Analyse en Composantes Principales (Chapitre 8)
- Implémentation complète de l'ACP
- Calcul des composantes principales
- Analyse de la variance expliquée
- Décomposition SVD

### 4. Régression Linéaire (Chapitre 9)
- Régression linéaire multidimensionnelle
- Estimateur des moindres carrés
- Calcul des intervalles de confiance
- Analyse des garanties statistiques

## Utilisation

Le script peut être exécuté directement :

```python
python main.py
```

Les résultats incluront :
- Des démonstrations sur des données synthétiques
- Des métriques de performance
- Des visualisations des résultats

## Dépendances

- Python >= 3.6
- NumPy: Pour les calculs matriciels
- SciPy.stats: Pour les calculs statistiques (distribution t)

## Exemples

### Classification avec LDA

```python
lda = LDA_Scratch()
lda.fit(X_train, y_train)
predictions = lda.predict(X_test)
```

### Analyse en Composantes Principales

```python
# Centrer les données
X_centered = X - np.mean(X, axis=0)

# Calculer la matrice de covariance
S = np.cov(X_centered.T)

# Décomposition en valeurs propres
eigenvalues, eigenvectors = np.linalg.eig(S)
```

### Régression Linéaire

```python
# Ajouter une colonne de 1 pour l'intercept
X = np.hstack((np.ones((n_samples, 1)), X_orig))

# Calculer les coefficients
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
```

---
**Note**: Les implémentations sont volontairement "from scratch" pour des fins pédagogiques. Pour une utilisation en production, privilégier des bibliothèques optimisées comme scikit-learn.
