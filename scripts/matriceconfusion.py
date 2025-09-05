import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Tes résultats
TP, TN, FP, FN = 78, 3040, 1796, 36

# Matrice 2x2
cm = np.array([[TP, FN],
               [FP, TN]])

# Labels des lignes/colonnes
labels = ["Positif (même sujet)", "Négatif (différent sujet)"]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Prédit Positif", "Prédit Négatif"],
            yticklabels=["Réel Positif", "Réel Négatif"])

plt.title("Matrice de confusion")
plt.ylabel("Classe réelle")
plt.xlabel("Classe prédite")
plt.tight_layout()
plt.show()
