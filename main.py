import numpy as np
import scipy.stats  # Pour la distribution t dans les intervalles de confiance de la régression linéaire


# --- Fonctions Utilitaires ---
def multivariate_normal_pdf(x, mean, covariance):
    """
    Calcule la fonction de densité de probabilité (PDF) d'une distribution normale multivariée.
    x: point de données (tableau 1D ou liste)
    mean: vecteur moyen (tableau 1D)
    covariance: matrice de covariance (tableau 2D)
    """
    k = len(mean)
    if k == 0: return 0  # Ne devrait pas arriver avec des données correctes

    # Assurer que la covariance est définie positive et bien conditionnée pour l'inversion
    # cov_stable = covariance + np.eye(k) * 1e-9 # Ajout d'un petit epsilon pour la stabilité numérique si besoin
    cov_stable = covariance

    try:
        det_cov = np.linalg.det(cov_stable)
        if det_cov <= 1e-100:  # Vérifier un déterminant non positif ou très proche de zéro
            # print(f"Attention: Déterminant de la covariance non-positif ou nul ({det_cov}).")
            return 1e-100  # Retourner un nombre très petit

        inv_cov = np.linalg.inv(cov_stable)
    except np.linalg.LinAlgError:
        # print("Attention: Matrice de covariance singulière dans multivariate_normal_pdf.")
        return 1e-100  # Si singulière, retourner un nombre très petit

    part1 = 1 / (((2 * np.pi) ** k * det_cov) ** 0.5)
    diff = (x - mean).reshape(-1, 1)  # Assurer un vecteur colonne
    part2 = (-0.5) * (diff.T @ inv_cov @ diff)

    pdf_val = part1 * np.exp(part2.item())  # .item() pour obtenir un scalaire d'une matrice 1x1
    return max(pdf_val, 1e-100)  # Éviter de retourner exactement zéro pour les logs ultérieurs


def univariate_normal_pdf(x, mean, variance):
    """PDF pour une distribution normale univariée."""
    if variance <= 1e-100:  # Éviter division par zéro ou racine de négatif
        return 1e-100
    return (1.0 / (np.sqrt(2 * np.pi * variance))) * np.exp(-0.5 * ((x - mean) ** 2 / variance))


# --- Chapitre 7: Apprentissage supervisé : classification ---
print("--- Chapitre 7: Classification ---")

# Génération de données synthétiques (identiques à avant pour la cohérence)
np.random.seed(42)
mean0_true = np.array([0, 0])
cov0_true = np.array([[1, 0.5], [0.5, 1]])
prior0_true_val = 0.5  # P(Y=0) vrai

mean1_true = np.array([3, 3])
cov1_true = np.array([[1, -0.5], [-0.5, 1]])
prior1_true_val = 0.5  # P(Y=1) vrai

X0_sample = np.random.multivariate_normal(mean0_true, cov0_true, size=100)
X1_sample = np.random.multivariate_normal(mean1_true, cov1_true, size=100)
X_data_class = np.vstack((X0_sample, X1_sample))
y_data_class = np.array([0] * 100 + [1] * 100)

# Pour tester, séparons en ensembles d'entraînement et de test
indices = np.arange(X_data_class.shape[0])
np.random.shuffle(indices)
split_idx = int(0.7 * X_data_class.shape[0])
train_indices, test_indices = indices[:split_idx], indices[split_idx:]

X_train_class, y_train_class = X_data_class[train_indices], y_data_class[train_indices]
X_test_class, y_test_class = X_data_class[test_indices], y_data_class[test_indices]

# 7.2 Classifieur de Bayes (Théorique) - Déjà "from scratch" conceptuellement
print("\n=== 7.2 Classifieur de Bayes (Théorique) ===")


# Ceci suppose que nous *connaissons* les vrais paramètres (mean0_true, cov0_true, prior0_true_val, etc.)

def bayes_classifier_theoretical_predict_one(x_point, m0, c0, p0, m1, c1, p1):
    # P(X=x|Y=0) * P(Y=0)
    posterior_0 = multivariate_normal_pdf(x_point, m0, c0) * p0
    # P(X=x|Y=1) * P(Y=1)
    posterior_1 = multivariate_normal_pdf(x_point, m1, c1) * p1

    if posterior_1 > posterior_0:
        return 1  # Prédire classe 1
    else:
        return 0  # Prédire classe 0


test_point_class = np.array([1.5, 1.5])
prediction_bayes_th = bayes_classifier_theoretical_predict_one(test_point_class,
                                                               mean0_true, cov0_true, prior0_true_val,
                                                               mean1_true, cov1_true, prior1_true_val)
print(f"Prédiction du classifieur de Bayes théorique pour {test_point_class}: {prediction_bayes_th}")

# 7.3 Plug-in classifieur (Estimation des paramètres à partir des données)
print("\n=== 7.3 Plug-in classifieur (Implémentation Manuelle) ===")


# Basé sur la définition : "un estimateur de n_i(X)" (où n_i(X) = P(Y=i|X))
# Nous estimons P(Y=i) (les priors) et P(X|Y=i) (les vraisemblances conditionnelles par classe)

class PluginClassifier:  # Agit comme QDA si on suppose des vraisemblances Gaussiennes
    def __init__(self):
        self.priors_ = {}  # pi_hat_i: estimations des P(Y=i)
        self.means_ = {}  # mu_hat_i: estimations des E[X|Y=i]
        self.covariances_ = {}  # Sigma_hat_i: estimations des Cov(X|Y=i)
        self.classes_ = None  # Liste des classes uniques

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples = len(y)

        for cls_val in self.classes_:
            X_cls = X[y == cls_val]
            self.priors_[cls_val] = len(X_cls) / n_samples  # Estimer P(Y=i) (Page 49, étape 1)
            self.means_[cls_val] = np.mean(X_cls, axis=0)  # Estimer E[X|Y=i]
            if X_cls.shape[0] > 1:  # Il faut au moins 2 échantillons pour la covariance
                if X_cls.shape[1] == 1:  # Cas d'une seule feature
                    self.covariances_[cls_val] = np.atleast_2d(
                        np.cov(X_cls.flatten(), ddof=0))  # ddof=0 pour MLE (1/N_i)
                else:
                    self.covariances_[cls_val] = np.cov(X_cls.T, ddof=0)  # ddof=0 pour MLE (1/N_i) (Prop. 7.4.3)
            else:  # Gérer les cas avec peu d'échantillons par classe
                self.covariances_[cls_val] = np.eye(X.shape[1]) * 1e-6  # Petite matrice identité

    def predict_one(self, x):
        posteriors = []
        for cls_val in self.classes_:
            prior = self.priors_[cls_val]
            mean = self.means_[cls_val]
            cov = self.covariances_[cls_val]

            # P_hat(X=x|Y=cls_val) * P_hat(Y=cls_val)
            likelihood = multivariate_normal_pdf(x, mean, cov)
            posterior = likelihood * prior
            posteriors.append(posterior)

        # Théorème 7.2.1 (généralisé pour K classes): choisir la classe avec la plus grande probabilité a posteriori
        return self.classes_[np.argmax(posteriors)]

    def predict(self, X_set):
        return np.array([self.predict_one(x) for x in X_set])


plugin_clf = PluginClassifier()
plugin_clf.fit(X_train_class, y_train_class)
y_pred_plugin = plugin_clf.predict(X_test_class)
accuracy_plugin = np.mean(y_pred_plugin == y_test_class)
print(f"Précision du Plug-in Classifieur (type QDA): {accuracy_plugin:.4f}")
print(f"Prédiction du Plug-in pour {test_point_class}: {plugin_clf.predict_one(test_point_class)}")

# 7.4 Analyse discriminante : LDA et QDA (Implémentation Manuelle)
print("\n=== 7.4 LDA et QDA (Implémentation Manuelle) ===")


# QDA (Analyse Discriminante Quadratique) est essentiellement ce que PluginClassifier ci-dessus implémente.
# Le théorème 7.4.2 donne la frontière de décision pour QDA pour 2 classes.
# Pour la prédiction, on utilise la règle de Bayes générale avec les estimations Gaussiennes spécifiques à chaque classe.

class LDA_Scratch:
    def __init__(self):
        self.priors_ = {}
        self.means_ = {}
        self.pooled_covariance_ = None  # Sigma (unique pour toutes les classes)
        self.inv_pooled_covariance_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        # Estimation des priors et des moyennes (identique à QDA/Plugin)
        class_sum_sq_dev = np.zeros((n_features, n_features))

        for cls_val in self.classes_:
            X_cls = X[y == cls_val]
            n_cls = X_cls.shape[0]
            self.priors_[cls_val] = n_cls / n_samples
            mean_cls = np.mean(X_cls, axis=0)
            self.means_[cls_val] = mean_cls

            # Somme des matrices de déviation carrée pour la covariance groupée
            if n_cls > 1:
                # (X_cls - mean_cls).T @ (X_cls - mean_cls)
                # ou sum (x_j - mu_i)(x_j - mu_i)^T  = (n_i - 1) * S_i_sample_cov
                # S_i_sample_cov est np.cov(X_cls.T, ddof=1)
                class_sum_sq_dev += (n_cls - 1) * np.cov(X_cls.T, ddof=1) if n_features > 1 else (
                                                                                                             n_cls - 1) * np.atleast_2d(
                    np.var(X_cls, ddof=1))

        # Calculer la covariance groupée (pooled covariance)
        # Sigma_pooled = (1 / (N - K)) * sum_i (n_i - 1) * S_i_sample_cov
        # où K est le nombre de classes
        if n_samples > len(self.classes_):
            self.pooled_covariance_ = class_sum_sq_dev / (n_samples - len(self.classes_))
        else:  # Fallback si pas assez d'échantillons
            self.pooled_covariance_ = np.eye(n_features) * 1e-6

        # Assurer que la matrice n'est pas singulière
        if np.linalg.det(self.pooled_covariance_) < 1e-100:  # Proche de zéro
            self.pooled_covariance_ += np.eye(n_features) * 1e-6  # Régularisation

        try:
            self.inv_pooled_covariance_ = np.linalg.inv(self.pooled_covariance_)
        except np.linalg.LinAlgError:
            # print("Attention: Pooled covariance singulière pour LDA.")
            self.inv_pooled_covariance_ = np.linalg.pinv(self.pooled_covariance_)

    def predict_one(self, x):
        # Règle de décision du Théorème 7.4.1 (pour 2 classes 0 et 1, page 50)
        # f*(x) = 1 si (mu1-mu0)^T Sigma^-1 (x - (mu1+mu0)/2) >= log(pi0/pi1)
        # Pour K classes, on calcule les fonctions discriminantes linéaires (scores)
        # delta_k(x) = x^T Sigma^-1 mu_k - 0.5 * mu_k^T Sigma^-1 mu_k + log(pi_k)

        scores = []
        for cls_val in self.classes_:
            mu_k = self.means_[cls_val]
            pi_k = self.priors_[cls_val]

            term1 = x.T @ self.inv_pooled_covariance_ @ mu_k
            term2 = -0.5 * mu_k.T @ self.inv_pooled_covariance_ @ mu_k
            term3 = np.log(pi_k) if pi_k > 1e-100 else -np.inf  # Éviter log(0)
            score = term1 + term2 + term3
            scores.append(score)

        return self.classes_[np.argmax(scores)]

    def predict(self, X_set):
        return np.array([self.predict_one(x) for x in X_set])


lda_scratch = LDA_Scratch()
lda_scratch.fit(X_train_class, y_train_class)
y_pred_lda_scratch = lda_scratch.predict(X_test_class)
accuracy_lda_scratch = np.mean(y_pred_lda_scratch == y_test_class)
print(f"Précision LDA (Implémentation Manuelle): {accuracy_lda_scratch:.4f}")
print(f"Prédiction LDA (Manuelle) pour {test_point_class}: {lda_scratch.predict_one(test_point_class)}")

print(f"Précision QDA (Manuelle via PluginClassifier): {accuracy_plugin:.4f}")  # Rappel

# 7.5 Bayes naïf (Gaussien, Implémentation Manuelle)
print("\n=== 7.5 Bayes naïf (Gaussien, Implémentation Manuelle) ===")


# Suppose que les features X_j sont conditionnellement indépendantes étant donné la classe Y.
# P(X|Y=i) = produit_j P(X_j | Y=i)
# Pour Bayes Naïf Gaussien, P(X_j | Y=i) ~ N(mu_ij, sigma_sq_ij) (Page 53)

class GaussianNaiveBayes_Scratch:
    def __init__(self):
        self.priors_ = {}  # pi_hat_i: estimations des P(Y=i)
        self.means_ = {}  # mu_hat_ij (dict de dicts: classe -> index_feature -> moyenne)
        self.variances_ = {}  # sigma_sq_hat_ij (dict de dicts: classe -> index_feature -> variance)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        for cls_val in self.classes_:
            X_cls = X[y == cls_val]
            self.priors_[cls_val] = X_cls.shape[0] / n_samples  # Estimation de pi_i (Page 53, étape 1)

            self.means_[cls_val] = {}
            self.variances_[cls_val] = {}
            for j in range(n_features):  # Pour chaque feature X^j
                self.means_[cls_val][j] = np.mean(X_cls[:, j])  # mu_ij
                self.variances_[cls_val][j] = np.var(X_cls[:, j], ddof=0)  # sigma_ij^2 (MLE, 1/N_i)
                if self.variances_[cls_val][j] < 1e-9:  # Éviter une variance nulle
                    self.variances_[cls_val][j] = 1e-9

    def predict_one(self, x):
        log_posteriors = []  # Travailler avec les log-probabilités pour éviter l'underflow
        n_features = len(x)
        for cls_val in self.classes_:
            log_prior = np.log(self.priors_[cls_val]) if self.priors_[cls_val] > 1e-100 else -np.inf

            log_sum_likelihoods = 0
            for j in range(n_features):  # produit_j P(X_j | Y=i) -> sum_j log P(X_j | Y=i)
                mean_ij = self.means_[cls_val][j]
                var_ij = self.variances_[cls_val][j]
                # P(X^j=x_j | Y=i) avec PDF univariée normale
                pdf_val = univariate_normal_pdf(x[j], mean_ij, var_ij)
                log_sum_likelihoods += np.log(pdf_val + 1e-100)  # +epsilon pour éviter log(0)

            log_posterior = log_prior + log_sum_likelihoods
            log_posteriors.append(log_posterior)

        return self.classes_[np.argmax(log_posteriors)]

    def predict(self, X_set):
        return np.array([self.predict_one(x) for x in X_set])


gnb_scratch = GaussianNaiveBayes_Scratch()
gnb_scratch.fit(X_train_class, y_train_class)
y_pred_gnb_scratch = gnb_scratch.predict(X_test_class)
accuracy_gnb_scratch = np.mean(y_pred_gnb_scratch == y_test_class)
print(f"Précision Bayes Naïf Gaussien (Manuelle): {accuracy_gnb_scratch:.4f}")
print(f"Prédiction GNB (Manuelle) pour {test_point_class}: {gnb_scratch.predict_one(test_point_class)}")

# --- Chapitre 8: Analyse en composantes principales (ACP) ---
print("\n\n--- Chapitre 8: Analyse en Composantes Principales (Implémentation Manuelle) ---")
# Génération de données 3D corrélées (identiques à avant)
np.random.seed(42)
mean_pca_data = np.array([1, 2, 3])  # Moyenne non nulle pour tester le centrage
cov_pca_data = np.array([[5, 2, 0.5], [2, 3, 1], [0.5, 1, 1]])
X_pca_orig = np.random.multivariate_normal(mean_pca_data, cov_pca_data, 100)

# 1. Centrer les données (Page 57: "données sont centrées")
mean_X = np.mean(X_pca_orig, axis=0)
X_pca_centered = X_pca_orig - mean_X
print(
    f"Données centrées. Moyenne originale: {mean_X.round(2)}, Nouvelle moyenne (devrait être ~0): {np.mean(X_pca_centered, axis=0).round(2)}")

# 2. Calculer la matrice de covariance empirique S (Page 57)
# S = (1/n) X_centré^T X_centré (MLE) ou (1/(n-1)) pour covariance d'échantillon.
# np.cov utilise (n-1) par défaut.
n_samples_pca = X_pca_centered.shape[0]
# S_empirical = (1/(n_samples_pca - 1)) * (X_pca_centered.T @ X_pca_centered) # Identique à np.cov(..., ddof=1)
S_empirical = np.cov(X_pca_centered.T, ddof=1)  # ddof=1 pour (n-1)

print(f"\nMatrice de Covariance Empirique S (forme {S_empirical.shape}):\n{S_empirical.round(2)}")

# 3. Décomposition en valeurs propres de S (Pages 56, 58: S = P Lambda P^T)
eigenvalues, eigenvectors = np.linalg.eig(S_empirical)

# Trier les valeurs propres et les vecteurs propres correspondants par ordre décroissant
sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices pour trier
eigenvalues_sorted = eigenvalues[sorted_indices]  # lambda_1 >= lambda_2 >= ...
eigenvectors_sorted = eigenvectors[:, sorted_indices]  # Vecteurs propres (p_i) en colonnes

print(f"\nValeurs Propres triées (lambda_i):\n{eigenvalues_sorted.round(2)}")
print(f"Vecteurs Propres triés (p_i, directions principales, en colonnes):\n{eigenvectors_sorted.round(2)}")

# 4. Choisir le nombre de composantes (ex: k=2) et former P_k (matrice des k premiers vecteurs propres)
k_components = 2
P_k = eigenvectors_sorted[:, :k_components]

# 5. Projeter les données sur le nouveau sous-espace de k dimensions: X_ACP = X_centré @ P_k (Page 62)
X_pca_projected = X_pca_centered @ P_k
print(f"\nDonnées projetées (5 premières lignes, {k_components} composantes):\n{X_pca_projected[:5].round(2)}")

# Part de la variance expliquée (Définition 8.3.2, page 59)
total_variance = np.sum(eigenvalues_sorted)  # tr(S)
explained_variance_ratio = eigenvalues_sorted / total_variance  # Pour chaque PC
cumulative_explained_variance = np.cumsum(explained_variance_ratio)  # Cumulée

print(f"\nRatio de variance expliquée par chaque CP:\n{explained_variance_ratio.round(2)}")
print(f"Variance expliquée cumulée:\n{cumulative_explained_variance.round(2)}")
for j_comp in range(1, len(eigenvalues_sorted) + 1):
    var_explained_j = np.sum(eigenvalues_sorted[:j_comp]) / total_variance
    print(f"Part de variance expliquée par les {j_comp} première(s) CP(s): {var_explained_j:.4f}")

# Perspective SVD (Page 61, Proposition 8.4.1, Théorème 8.4.3)
# X_centré = U Sigma_svd V^T
# Les colonnes de V sont les vecteurs propres de X_centré^T X_centré
# ( (n-1)*S_empirique = X_centré^T X_centré )
U_svd, s_values_svd, Vt_svd = np.linalg.svd(X_pca_centered, full_matrices=False)
V_svd = Vt_svd.T  # Les vecteurs singuliers droits sont les colonnes de V
print(f"\nPremière CP via SVD de X_centré (V_svd[:,0]):\n{V_svd[:, 0].round(2)}")
print(f"Première CP via Décomp. Eig. de S (eigenvectors_sorted[:,0]):\n{eigenvectors_sorted[:, 0].round(2)}")
# Note: Les signes peuvent être inversés, mais ils couvrent la même direction.

# Les valeurs singulières au carré (s_values_svd^2) divisées par (n-1) devraient être les valeurs propres de S
print(f"\nValeurs propres de S: {eigenvalues_sorted.round(2)}")
print(f"Valeurs singulières de X_centré au carré / (n-1): {((s_values_svd ** 2) / (n_samples_pca - 1)).round(2)}")

# --- Chapitre 9: Apprentissage supervisé : régression linéaire ---
print("\n\n--- Chapitre 9: Régression Linéaire (Multidimensionnelle, Implémentation Manuelle) ---")
# Génération de données synthétiques (identiques à avant)
np.random.seed(42)
n_samples_reg, n_features_reg = 100, 3
X_reg_orig = np.random.rand(n_samples_reg, n_features_reg) * 10
true_beta_star = np.array([2.5, -1.0, 0.5])  # Vrais coefficients pour X1, X2, X3
true_alpha_star = 5.0  # Vrai intercept (ordonnée à l'origine)
noise_std_reg = 2.0
noise_reg = np.random.normal(0, noise_std_reg, n_samples_reg)
y_reg = true_alpha_star + X_reg_orig @ true_beta_star + noise_reg  # Modèle Y = alpha* + X beta* + bruit

# Ajouter une colonne de uns à X pour l'intercept (Page 68, "Yi = X_tilde_i beta_tilde* + xi_i")
# beta_tilde* = (alpha*, beta_1*, ..., beta_d*)^T
# X_tilde_i = (1, X_i1, ..., X_id)
X_reg_intercept = np.hstack((np.ones((n_samples_reg, 1)), X_reg_orig))
d_params = X_reg_intercept.shape[1]  # Nombre de paramètres (d features originales + 1 intercept)

# Théorème 9.2.1: beta_hat = (X^T X)^-1 X^T Y (Page 68)
# Ce beta_hat (que j'appelle beta_hat_full) inclut l'intercept comme sa première composante.
try:
    XTX = X_reg_intercept.T @ X_reg_intercept  # X_tilde^T X_tilde
    XTX_inv = np.linalg.inv(XTX)
    XTY = X_reg_intercept.T @ y_reg  # X_tilde^T Y
    beta_hat_full = XTX_inv @ XTY  # Estimateur des moindres carrés

    alpha_hat_reg = beta_hat_full[0]  # Estimation de l'intercept
    beta_coeffs_reg = beta_hat_full[1:]  # Estimations des coefficients de pente

    print(f"\nEstimations des Moindres Carrés (Manuelles):")
    print(f"  alpha_hat (intercept): {alpha_hat_reg:.4f} (Vrai: {true_alpha_star})")
    print(f"  beta_hat (coefficients): {beta_coeffs_reg.round(4)} (Vrais: {true_beta_star})")
except np.linalg.LinAlgError:
    print("Erreur: Matrice X^T X singulière. Impossible de calculer l'inverse.")
    beta_hat_full = None

# Prédictions
if beta_hat_full is not None:
    y_pred_reg = X_reg_intercept @ beta_hat_full  # Y_hat = X_tilde beta_hat_full

    # Coefficient de Détermination (R^2) - Définition 9.1.2 (Page 67)
    # r^2 = (variabilité expliquée par le modèle) / (variabilité totale des observations)
    # r^2 = 1 - sum( (Y_i - Y_hat_i)^2 ) / sum( (Y_i - Y_moyen)^2 )
    # r^2 = 1 - SS_res / SS_tot

    ss_residuals = np.sum((y_reg - y_pred_reg) ** 2)  # Somme des carrés des résidus
    ss_total = np.sum((y_reg - np.mean(y_reg)) ** 2)  # Somme totale des carrés
    if ss_total < 1e-100:  # Éviter division par zéro si tous les y_reg sont identiques
        r_squared_reg = 1.0 if ss_residuals < 1e-100 else 0.0
    else:
        r_squared_reg = 1 - (ss_residuals / ss_total)
    print(f"\nCoefficient de Détermination (R^2, Manuel): {r_squared_reg:.4f}")

    # 9.3 Garanties statistiques sur l'estimateur des moindres carrés
    print("\n--- 9.3 Garanties Statistiques (Implémentation Manuelle) ---")
    # Supposant un bruit Gaussien xi_i ~ N(0, sigma_star^2)

    # Estimation de sigma_star^2 (variance du bruit) - Théorème 9.3.3 (Page 72)
    # sigma_hat^2 = ||Y - X beta_hat||^2 / (n - d_params)
    # d_params est le nombre de paramètres dans beta_hat_full (i.e., rang de X_reg_intercept)
    sigma_sq_hat_reg = ss_residuals / (n_samples_reg - d_params)
    print(
        f"Variance estimée du bruit (sigma_sq_hat): {sigma_sq_hat_reg:.4f} (Vraie sigma_star^2: {noise_std_reg ** 2:.4f})")

    # Variance de beta_hat (coefficients incluant l'intercept) - Page 69 (Var[beta_hat] = sigma_star^2 * (X^T X)^-1)
    # On utilise sigma_sq_hat_reg comme estimation de sigma_star^2
    # XTX_inv a déjà été calculé.
    var_beta_hat_reg = sigma_sq_hat_reg * XTX_inv

    # Les erreurs standard (Standard Errors) sont les racines carrées des éléments diagonaux de var_beta_hat_reg
    se_beta_hat_reg = np.sqrt(np.diag(var_beta_hat_reg))
    print(f"\nErreurs Standard pour beta_hat_full (incl. intercept):\n{se_beta_hat_reg.round(4)}")

    # Intervalles de Confiance pour beta_j - Théorème 9.3.4 (Page 73)
    # beta_hat_j +/- t_critique(n-d_params, alpha/2) * SE(beta_hat_j)
    # où SE(beta_hat_j) = sigma_hat * sqrt((X^T X)^-1_{jj})
    confidence_level_alpha = 0.05  # Pour un intervalle de confiance à 95%
    dof_reg = n_samples_reg - d_params  # Degrés de liberté pour la distribution t

    # s^{n-d}_{1-alpha/2} du texte page 74 est la valeur t critique
    t_critical_reg = scipy.stats.t.ppf(1 - confidence_level_alpha / 2, df=dof_reg)
    print(f"Valeur t-critique pour IC à 95% (ddl={dof_reg}): {t_critical_reg:.4f}")

    param_names_reg = ['alpha_hat (intercept)'] + [f'beta_hat_{i + 1}' for i in range(n_features_reg)]
    print("\nIntervalles de Confiance à 95% pour les paramètres (Manuels):")
    for i in range(d_params):
        lower_bound = beta_hat_full[i] - t_critical_reg * se_beta_hat_reg[i]
        upper_bound = beta_hat_full[i] + t_critical_reg * se_beta_hat_reg[i]
        print(f"  {param_names_reg[i]:<22}: [{lower_bound:.4f}, {upper_bound:.4f}]")
        # Vérifier si le vrai paramètre est dans l'IC (pour notre exemple synthétique)
        true_param_val = true_alpha_star if i == 0 else true_beta_star[i - 1]
        if lower_bound <= true_param_val <= upper_bound:
            print(f"    Le vrai paramètre ({true_param_val:.2f}) est dans l'IC.")
        else:
            print(f"    ATTENTION: Le vrai paramètre ({true_param_val:.2f}) N'EST PAS dans l'IC.")
else:
    print("Garanties statistiques de la régression linéaire sautées à cause d'une erreur précédente.")

print("\n\nFin des implémentations 'Manuelles'.")