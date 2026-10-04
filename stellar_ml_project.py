# -*- coding: utf-8 -*-
"""
Machine Learning Stellar Evolution from Surface Observables

Objetivos:
1) Predecir masa estelar con Random Forest y Gradient Boosting.
2) Medir cuánto aportan [C/Fe] y [N/Fe] a la predicción de masa.
3) Clasificar RGB vs RC con una red neuronal (MLP).
4) Propagar incertidumbres observacionales con Monte Carlo y bootstrap.

El script descarga APOKASC-3 desde VizieR y guarda las figuras en ./figures/.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# -----------------------------------------------------------------------------
# 0. Configuración general
# -----------------------------------------------------------------------------

RANDOM_STATE = 42
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Variables superficiales que usaremos como entrada en ambos problemas.
features = [
    "Teff",
    "loggSpec",
    "[Fe/H]",
    "[a/Fe]",
    "[C/Fe]",
    "[N/Fe]",
]

# Incertidumbres observacionales asociadas a esas seis variables.
feature_errors = [
    "e_Teff",
    "e_loggSpec",
    "e_[Fe/H]",
    "e_[a/Fe]",
    "e_[C/Fe]",
    "e_[N/Fe]",
]


# -----------------------------------------------------------------------------
# 1. Descargar y seleccionar los datos de APOKASC-3
# -----------------------------------------------------------------------------

# ROW_LIMIT=-1 pide a VizieR todas las filas disponibles del catálogo.
Vizier.ROW_LIMIT = -1
catalog = Vizier.get_catalogs("J/ApJS/276/69/table4")[0]
df = catalog.to_pandas()

columns = [
    "KIC",
    "EvolSt",
    "EvolSrc",
    "Teff",
    "e_Teff",
    "loggSpec",
    "e_loggSpec",
    "[Fe/H]",
    "e_[Fe/H]",
    "[a/Fe]",
    "e_[a/Fe]",
    "[C/Fe]",
    "e_[C/Fe]",
    "[N/Fe]",
    "e_[N/Fe]",
    "Mass",
    "e_Mass",
]

data = df[columns].copy()

print("Catálogo completo:", data.shape)
print("\nEstados evolutivos:")
print(data["EvolSt"].value_counts(dropna=False))
print("\nOrigen del estado evolutivo:")
print(data["EvolSrc"].value_counts(dropna=False))


# -----------------------------------------------------------------------------
# 2. Preparar los dos conjuntos de trabajo
# -----------------------------------------------------------------------------

# CLASIFICACIÓN:
# Solo usamos etiquetas RGB/RC cuyo origen sea sísmico (EvolSrc='Seis').
# Así evitamos entrenar la red con etiquetas puramente espectroscópicas.
class_data = data[
    (data["EvolSrc"] == "Seis")
    & (data["EvolSt"].isin(["RGB", "RC"]))
].copy()
class_data = class_data.dropna(subset=features)

X_class = class_data[features].values
y_class = class_data["EvolSt"].values
X_err_class = class_data[feature_errors].values

print("\nMuestra de clasificación:", class_data.shape)
print(class_data["EvolSt"].value_counts())

# REGRESIÓN:
# Conservamos RGB y RC con masa de catálogo y las seis variables disponibles.
reg_data = data[data["EvolSt"].isin(["RGB", "RC"])].copy()
reg_data = reg_data.dropna(subset=features + ["Mass", "e_Mass"])

X_reg = reg_data[features].values
y_reg = reg_data["Mass"].values
X_err_reg = reg_data[feature_errors].values
y_err_reg = reg_data["e_Mass"].values

print("Muestra de regresión:", reg_data.shape)


# -----------------------------------------------------------------------------
# 3. Regresión: separar TRAIN y TEST una sola vez
# -----------------------------------------------------------------------------

# Pasamos también los errores al split para que cada incertidumbre permanezca
# alineada con la misma estrella que sus features y su masa.
(
    X_train_reg,
    X_test_reg,
    y_train_reg,
    y_test_reg,
    X_err_train,
    X_err_test,
    y_err_train,
    y_err_test,
) = train_test_split(
    X_reg,
    y_reg,
    X_err_reg,
    y_err_reg,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

print("\nRegresión TRAIN:", X_train_reg.shape)
print("Regresión TEST:", X_test_reg.shape)


# -----------------------------------------------------------------------------
# 4. Selección de modelo por validación cruzada
# -----------------------------------------------------------------------------

# KFold divide solo el TRAIN en 5 partes. El TEST sigue sin intervenir en la
# elección de hiperparámetros.
cv_reg = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rf_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 8, 12, None],
    "min_samples_leaf": [1, 2, 4],
}

rf_search = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    rf_grid,
    cv=cv_reg,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
rf_search.fit(X_train_reg, y_train_reg)

# Gradient Boosting se usa como segundo modelo para comparar si el RF realmente
# es la mejor opción para estos datos.
gb_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [1, 2, 3],
}

gb_search = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    gb_grid,
    cv=cv_reg,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
gb_search.fit(X_train_reg, y_train_reg)

print("\nMejores parámetros RF:", rf_search.best_params_)
print("MAE CV RF:", -rf_search.best_score_)
print("Mejores parámetros GB:", gb_search.best_params_)
print("MAE CV GB:", -gb_search.best_score_)


# -----------------------------------------------------------------------------
# 5. Evaluación final de regresión en TEST
# -----------------------------------------------------------------------------

best_rf = rf_search.best_estimator_
best_gb = gb_search.best_estimator_

y_pred_rf = best_rf.predict(X_test_reg)
y_pred_gb = best_gb.predict(X_test_reg)

mae_rf = mean_absolute_error(y_test_reg, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf))
r2_rf = r2_score(y_test_reg, y_pred_rf)

mae_gb = mean_absolute_error(y_test_reg, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test_reg, y_pred_gb))
r2_gb = r2_score(y_test_reg, y_pred_gb)

print("\nRANDOM FOREST")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R2:", r2_rf)

print("\nGRADIENT BOOSTING")
print("MAE:", mae_gb)
print("RMSE:", rmse_gb)
print("R2:", r2_gb)

# Figura 1: masa real/catalogada frente a masa predicha.
plt.figure(figsize=(6, 6))
plt.scatter(y_test_reg, y_pred_rf, s=12, alpha=0.4)
minimum = min(y_test_reg.min(), y_pred_rf.min())
maximum = max(y_test_reg.max(), y_pred_rf.max())
plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
plt.xlabel("Seismic mass [M_sun]")
plt.ylabel("Predicted mass [M_sun]")
plt.title("Random Forest: seismic vs predicted mass")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_rf_seismic_vs_predicted.png", dpi=200)
plt.show()


# -----------------------------------------------------------------------------
# 6. ¿Qué variables usa más el Random Forest?
# -----------------------------------------------------------------------------

# feature_importances_ cuantifica cuánto contribuyó cada variable a reducir el
# error dentro de los árboles. Es importancia predictiva, no causalidad física.
importances = best_rf.feature_importances_

print("\nFeature importances:")
for name, importance in zip(features, importances):
    print(f"{name}: {importance:.6f}")

plt.figure(figsize=(7, 4))
plt.bar(features, importances)
plt.ylabel("Feature importance")
plt.title("Random Forest feature importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_rf_feature_importance.png", dpi=200)
plt.show()


# -----------------------------------------------------------------------------
# 7. Ablation test: repetir la regresión sin C y N
# -----------------------------------------------------------------------------

# Las cuatro primeras columnas son Teff, loggSpec, [Fe/H] y [a/Fe].
# Mantener el mismo TRAIN/TEST permite aislar el efecto de retirar C y N.
X_train_no_cn = X_train_reg[:, :4]
X_test_no_cn = X_test_reg[:, :4]

rf_grid_no_cn = {
    "n_estimators": [200],
    "max_depth": [8, 12, None],
    "min_samples_leaf": [2, 4, 8],
}

rf_search_no_cn = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    rf_grid_no_cn,
    cv=cv_reg,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
rf_search_no_cn.fit(X_train_no_cn, y_train_reg)

rf_no_cn = rf_search_no_cn.best_estimator_
y_pred_no_cn = rf_no_cn.predict(X_test_no_cn)

mae_no_cn = mean_absolute_error(y_test_reg, y_pred_no_cn)
rmse_no_cn = np.sqrt(mean_squared_error(y_test_reg, y_pred_no_cn))
r2_no_cn = r2_score(y_test_reg, y_pred_no_cn)

print("\nRF SIN C/N")
print("Mejores parámetros:", rf_search_no_cn.best_params_)
print("MAE CV:", -rf_search_no_cn.best_score_)
print("MAE TEST:", mae_no_cn)
print("RMSE TEST:", rmse_no_cn)
print("R2 TEST:", r2_no_cn)

plt.figure(figsize=(6, 4))
plt.bar(["RF with C/N", "RF without C/N"], [mae_rf, mae_no_cn])
plt.ylabel("MAE [M_sun]")
plt.title("Impact of C and N on stellar mass prediction")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_cn_ablation_regression.png", dpi=200)
plt.show()


# -----------------------------------------------------------------------------
# 8. Residuales: buscar sesgos sistemáticos del modelo
# -----------------------------------------------------------------------------

# Residual positivo -> el RF sobreestima la masa.
# Residual negativo -> el RF infraestima la masa.
residuals = y_pred_rf - y_test_reg

plt.figure(figsize=(7, 5))
plt.scatter(y_test_reg, residuals, s=12, alpha=0.4)
plt.axhline(0, linestyle="--")
plt.xlabel("Seismic mass [M_sun]")
plt.ylabel("Predicted - seismic mass [M_sun]")
plt.title("Random Forest residuals")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_rf_residuals.png", dpi=200)
plt.show()


# -----------------------------------------------------------------------------
# 9. Incertidumbre de masa: bootstrap + Monte Carlo
# -----------------------------------------------------------------------------

# Esta parte es la más lenta. Se usa una estrella de TEST con errores completos.
# El bootstrap cambia la muestra de entrenamiento; Monte Carlo perturba tanto
# las mediciones del TRAIN como las de la estrella nueva dentro de sus errores.
finite_train_unc = (
    np.all(np.isfinite(X_err_train), axis=1)
    & np.isfinite(y_err_train)
)
finite_test_unc = (
    np.all(np.isfinite(X_err_test), axis=1)
    & np.isfinite(y_err_test)
)

X_train_unc = X_train_reg[finite_train_unc]
y_train_unc = y_train_reg[finite_train_unc]
X_err_train_unc = X_err_train[finite_train_unc]
y_err_train_unc = y_err_train[finite_train_unc]

star_index_reg = np.where(finite_test_unc)[0][0]
star = X_test_reg[star_index_reg]
star_error = X_err_test[star_index_reg]
true_mass = y_test_reg[star_index_reg]
true_mass_error = y_err_test[star_index_reg]

normal_prediction = best_rf.predict(star.reshape(1, -1))[0]
print("\nPredicción RF nominal:", normal_prediction)
print("Masa de catálogo:", true_mass, "+/-", true_mass_error)

n_boot = 50
n_mc = 100
mass_predictions = []

for i in range(n_boot):
    # Semilla distinta, pero reproducible, en cada bootstrap.
    np.random.seed(i)

    # Muestreo con reemplazo del TRAIN.
    indices = np.random.choice(
        len(X_train_unc),
        size=len(X_train_unc),
        replace=True,
    )

    X_boot = X_train_unc[indices]
    y_boot = y_train_unc[indices]
    X_err_boot = X_err_train_unc[indices]
    y_err_boot = y_err_train_unc[indices]

    # Perturbar observables y masa objetivo dentro de sus errores catalogados.
    X_boot_perturbed = np.random.normal(X_boot, np.abs(X_err_boot))
    y_boot_perturbed = np.random.normal(y_boot, np.abs(y_err_boot))

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=4,
        random_state=i,
        n_jobs=-1,
    )
    model.fit(X_boot_perturbed, y_boot_perturbed)

    # 100 posibles observaciones de la misma estrella de TEST.
    star_samples = np.random.normal(
        star,
        np.abs(star_error),
        size=(n_mc, len(features)),
    )
    mass_predictions.extend(model.predict(star_samples))

mass_predictions = np.asarray(mass_predictions)

p16_mass, p50_mass, p84_mass = np.percentile(
    mass_predictions,
    [16, 50, 84],
)

print("Predicción final:", p50_mass)
print("Error inferior:", p50_mass - p16_mass)
print("Error superior:", p84_mass - p50_mass)

plt.figure(figsize=(7, 5))
plt.hist(mass_predictions, bins=40, density=True)
plt.axvline(p50_mass, linestyle="--", label="ML median")
plt.axvline(true_mass, linestyle=":", label="Seismic mass")
plt.xlabel("Predicted mass [M_sun]")
plt.ylabel("Density")
plt.title("ML mass uncertainty")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_mass_uncertainty.png", dpi=200)
plt.show()


# -----------------------------------------------------------------------------
# 10. Clasificación RGB vs RC con una red neuronal
# -----------------------------------------------------------------------------

# Incluimos los índices en el split para mantener alineados los errores de cada
# estrella con el conjunto TEST. random_state=123 crea el split final usado en
# esta rama del proyecto.
class_indices = np.arange(len(class_data))
(
    X_train_nn,
    X_test_nn,
    y_train_nn,
    y_test_nn,
    idx_train_nn,
    idx_test_nn,
) = train_test_split(
    X_class,
    y_class,
    class_indices,
    test_size=0.20,
    random_state=123,
    stratify=y_class,
)

# StandardScaler aprende media y desviación SOLO del TRAIN.
scaler_nn = StandardScaler()
X_train_nn_scaled = scaler_nn.fit_transform(X_train_nn)
X_test_nn_scaled = scaler_nn.transform(X_test_nn)

# Arquitectura fijada tras las pruebas exploratorias previas: dos capas ocultas
# de 16 y 8 neuronas. early_stopping reserva internamente parte del TRAIN para
# decidir cuándo detener el entrenamiento.
nn_final = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    alpha=0.01,
    learning_rate_init=0.001,
    max_iter=1500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)
nn_final.fit(X_train_nn_scaled, y_train_nn)

y_pred_nn = nn_final.predict(X_test_nn_scaled)
accuracy_nn = nn_final.score(X_test_nn_scaled, y_test_nn)

print("\nCLASIFICACIÓN RGB/RC")
print("Accuracy:", accuracy_nn)
print("Iteraciones:", nn_final.n_iter_)
print(classification_report(y_test_nn, y_pred_nn))
print("Matriz de confusión:")
print(confusion_matrix(y_test_nn, y_pred_nn, labels=["RC", "RGB"]))


# -----------------------------------------------------------------------------
# 11. Ablation test de clasificación: red sin C y N
# -----------------------------------------------------------------------------

X_train_no_cn_nn = X_train_nn[:, :4]
X_test_no_cn_nn = X_test_nn[:, :4]

# Se necesita un scaler nuevo porque ahora la red recibe 4 variables, no 6.
scaler_no_cn = StandardScaler()
X_train_no_cn_scaled = scaler_no_cn.fit_transform(X_train_no_cn_nn)
X_test_no_cn_scaled = scaler_no_cn.transform(X_test_no_cn_nn)

nn_no_cn = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    alpha=0.01,
    learning_rate_init=0.001,
    max_iter=1500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)
nn_no_cn.fit(X_train_no_cn_scaled, y_train_nn)

accuracy_no_cn = nn_no_cn.score(X_test_no_cn_scaled, y_test_nn)
y_pred_no_cn_nn = nn_no_cn.predict(X_test_no_cn_scaled)

print("\nNN CON C/N:", accuracy_nn)
print("NN SIN C/N:", accuracy_no_cn)
print("Matriz de confusión SIN C/N:")
print(confusion_matrix(y_test_nn, y_pred_no_cn_nn, labels=["RC", "RGB"]))


# -----------------------------------------------------------------------------
# 12. Incertidumbre de clasificación por errores observacionales
# -----------------------------------------------------------------------------

X_err_test_nn = X_err_class[idx_test_nn]
valid_errors = np.all(np.isfinite(X_err_test_nn), axis=1)

prob_test = nn_final.predict_proba(X_test_nn_scaled)

# Localizamos de forma explícita la columna que corresponde a RC. Así no
# dependemos de que sklearn ordene las clases de una forma concreta.
rc_column = np.where(nn_final.classes_ == "RC")[0][0]
p_rc_test = prob_test[:, rc_column]

# Elegimos una estrella cercana a la frontera P(RC)=0.5 y con errores completos.
distance = np.abs(p_rc_test - 0.5)
distance[~valid_errors] = np.inf
star_index_class = np.argmin(distance)

star_class = X_test_nn[star_index_class]
star_class_error = X_err_test_nn[star_index_class]

print("\nEstrella de frontera:")
print("Índice:", star_index_class)
print("Clase real:", y_test_nn[star_index_class])
print("P(RC) nominal:", p_rc_test[star_index_class])

# Generamos 5000 posibles observaciones de esa estrella dentro de sus errores.
np.random.seed(RANDOM_STATE)
n_mc_class = 5000
star_samples_class = np.random.normal(
    loc=star_class,
    scale=np.abs(star_class_error),
    size=(n_mc_class, len(features)),
)

star_samples_class_scaled = scaler_nn.transform(star_samples_class)
prob_mc = nn_final.predict_proba(star_samples_class_scaled)
p_rc = prob_mc[:, rc_column]

p16_rc, p50_rc, p84_rc = np.percentile(p_rc, [16, 50, 84])
fraction_rc = np.mean(p_rc > 0.5)

print("P(RC) mediana:", p50_rc)
print("Error inferior:", p50_rc - p16_rc)
print("Error superior:", p84_rc - p50_rc)
print("Fracción clasificada como RC:", fraction_rc)

plt.figure(figsize=(7, 5))
plt.hist(p_rc, bins=40, density=True)
plt.axvline(0.5, linestyle="--")
plt.xlabel("P(RC)")
plt.ylabel("Density")
plt.title("Classification uncertainty from observational errors")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_classification_uncertainty.png", dpi=200)
plt.show()


# -----------------------------------------------------------------------------
# 13. Resumen numérico final
# -----------------------------------------------------------------------------

print("\n================ RESUMEN ================")
print(f"RF: MAE={mae_rf:.4f}, RMSE={rmse_rf:.4f}, R2={r2_rf:.4f}")
print(f"GB: MAE={mae_gb:.4f}, RMSE={rmse_gb:.4f}, R2={r2_gb:.4f}")
print(f"RF sin C/N: MAE={mae_no_cn:.4f}, R2={r2_no_cn:.4f}")
print(f"NN RGB/RC: accuracy={accuracy_nn:.4f}")
print(f"NN sin C/N: accuracy={accuracy_no_cn:.4f}")
print("Figuras guardadas en:", FIGURES_DIR)
