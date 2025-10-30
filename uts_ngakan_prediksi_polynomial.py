# UTS Ngakan - Prediksi Harga Properti dengan Polynomial Regression

# ================================================================
# 0. Persiapan Lingkungan & Import Library
# ================================================================
# Jalankan ini dulu; install package bila perlu (di Colab: !pip install -q seaborn joblib)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (8,5)
sns.set(style='whitegrid')

# ================================================================
# 1. Data Preparation
# ================================================================
# Generate synthetic dataset (lebih dari 200 sampel)

np.random.seed(42)
n_samples = 1000  # > 200

luas_tanah = np.random.uniform(50, 500, n_samples)
luas_bangunan = np.random.uniform(30, 400, n_samples)
jml_kamar = np.random.randint(1, 6, n_samples)
umur = np.random.uniform(0, 30, n_samples)
jarak = np.random.uniform(1, 20, n_samples)

# Hubungan non-linear realistis
price_base = (
    0.8 * luas_tanah +
    1.2 * luas_bangunan +
    30 * jml_kamar -
    10 * umur -
    15 * jarak
)

price_nonlinear = price_base + 0.01*(luas_bangunan**2 / 100) - 0.005*(jarak**2) + 0.02*(luas_tanah*luas_bangunan/100)
noise = np.random.normal(0, 150, n_samples)
price_raw = price_nonlinear + noise
price = np.clip(price_raw, 200, 5000)

df = pd.DataFrame({
    'luas_tanah': luas_tanah,
    'luas_bangunan': luas_bangunan,
    'jml_kamar': jml_kamar,
    'umur': umur,
    'jarak': jarak,
    'harga': price
})

# Simpan dataset sintetis ke file CSV
df.to_csv('dataset_price.csv', index=False)
print("✅ Dataset berhasil disimpan sebagai dataset_price.csv")

df.head()

# ================================================================
# 1.B Exploratory Data Analysis (EDA)
# ================================================================

summary = df.describe().T
print(summary[['mean','std','min','max']])

df[['luas_tanah','luas_bangunan','jml_kamar','umur','jarak','harga']].hist(bins=30, figsize=(12,10))
plt.tight_layout()
plt.show()

# Scatter plot tiap fitur vs harga
features = ['luas_tanah','luas_bangunan','jml_kamar','umur','jarak']
fig, axs = plt.subplots(2,3, figsize=(15,10))
for ax, feat in zip(axs.flatten(), features+['harga'][:1]):
    if feat in features:
        ax.scatter(df[feat], df['harga'], alpha=0.4)
        ax.set_xlabel(feat)
        ax.set_ylabel('harga')
plt.tight_layout()
plt.show()

# Correlation matrix
corr = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Outlier detection sederhana
outlier_info = {}
for col in ['luas_tanah','luas_bangunan','harga']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR))
    outlier_info[col] = mask.sum()
print(outlier_info)

# ================================================================
# 1.C Data Preprocessing
# ================================================================

X = df.drop(columns='harga')
y = df['harga']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_price.pkl')
print("✅ Scaler berhasil disimpan sebagai scaler_price.pkl")

# ================================================================
# 2. Model Implementation
# ================================================================
from collections import defaultdict

models = defaultdict(dict)
degrees = [1,2,3,4,5]
alphas_ridge = [0.1, 1, 10]
alphas_lasso = [0.1, 1, 10]

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    Xtr_poly = poly.fit_transform(X_train_scaled)
    Xte_poly = poly.transform(X_test_scaled)

    lr = LinearRegression()
    lr.fit(Xtr_poly, y_train)
    models[d]['Linear'] = (lr, poly)

    for a in alphas_ridge:
        r = Ridge(alpha=a, max_iter=5000)
        r.fit(Xtr_poly, y_train)
        models[d][f'Ridge_a{a}'] = (r, poly)

    for a in alphas_lasso:
        l = Lasso(alpha=a, max_iter=5000)
        l.fit(Xtr_poly, y_train)
        models[d][f'Lasso_a{a}'] = (l, poly)

# ================================================================
# 3. Model Evaluation
# ================================================================

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

results = []
for d in degrees:
    for name, (model, poly) in models[d].items():
        Xtr = poly.transform(X_train_scaled)
        Xte = poly.transform(X_test_scaled)
        ytr_pred = model.predict(Xtr)
        yte_pred = model.predict(Xte)
        res = {
            'degree': d,
            'model': name,
            'train_r2': r2_score(y_train, ytr_pred),
            'test_r2': r2_score(y_test, yte_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, ytr_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, yte_pred)),
            'test_mape': mape(y_test.values, yte_pred)
        }
        results.append(res)

results_df = pd.DataFrame(results)
print(results_df.sort_values(['degree','model']).head(10))

# Visualisasi train vs test R2 per degree
agg = results_df.groupby(['degree']).apply(lambda g: g.loc[g['test_r2'].idxmax()]).reset_index(drop=True)
plt.plot(agg['degree'], agg['train_r2'], marker='o', label='Train R2')
plt.plot(agg['degree'], agg['test_r2'], marker='o', label='Test R2')
plt.title('Train vs Test R2 by Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('R2')
plt.legend()
plt.show()

# ================================================================
# 4. Regularization Analysis
# ================================================================

alpha_grid = [0.001, 0.01, 0.1, 1, 10, 100]
degree_for_reg = 4
poly_reg = PolynomialFeatures(degree=degree_for_reg, include_bias=False)
Xtr_poly = poly_reg.fit_transform(X_train_scaled)
Xte_poly = poly_reg.transform(X_test_scaled)

r_scores_ridge, r_scores_lasso = [], []
for a in alpha_grid:
    ridge = Ridge(alpha=a).fit(Xtr_poly, y_train)
    lasso = Lasso(alpha=a).fit(Xtr_poly, y_train)
    r_scores_ridge.append(ridge.score(Xte_poly, y_test))
    r_scores_lasso.append(lasso.score(Xte_poly, y_test))

plt.semilogx(alpha_grid, r_scores_ridge, marker='o', label='Ridge R2')
plt.semilogx(alpha_grid, r_scores_lasso, marker='o', label='Lasso R2')
plt.xlabel('alpha (log scale)')
plt.ylabel('R2 Test')
plt.title(f'Ridge vs Lasso R2 (Degree {degree_for_reg})')
plt.legend()
plt.show()

# ================================================================
# 5. Model Selection & Prediction
# ================================================================

# Cross-validation untuk pilih model terbaik
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
for d in degrees:
    for name, (model_obj, poly_obj) in models[d].items():
        X_all_poly = poly_obj.transform(scaler.transform(X))
        r2s, rmses = [], []
        for train_idx, val_idx in kf.split(X_all_poly):
            Xtr_fold, Xval_fold = X_all_poly[train_idx], X_all_poly[val_idx]
            ytr_fold, yval_fold = y.values[train_idx], y.values[val_idx]
            cls = model_obj.__class__
            params = model_obj.get_params()
            m = cls(**params)
            m.fit(Xtr_fold, ytr_fold)
            ypred = m.predict(Xval_fold)
            r2s.append(r2_score(yval_fold, ypred))
            rmses.append(np.sqrt(mean_squared_error(yval_fold, ypred)))
        cv_results.append({
            'degree': d,
            'model': name,
            'cv_r2_mean': np.mean(r2s),
            'cv_rmse_mean': np.mean(rmses)
        })

cv_df = pd.DataFrame(cv_results).sort_values(by='cv_r2_mean', ascending=False)
print(cv_df.head(5))

best_cv = cv_df.iloc[0]
best_degree = int(best_cv['degree'])
best_model_name = best_cv['model']
print(f"Best Model: Degree {best_degree}, {best_model_name}")

best_model, best_poly = models[best_degree][best_model_name]
joblib.dump({'model': best_model, 'poly': best_poly, 'scaler': scaler}, 'best_model_bundle.pkl')

# Prediksi data baru
residuals = y_test - best_model.predict(best_poly.transform(X_test_scaled))
resid_std = residuals.std()

def predict_property(input_df, model=best_model, poly=best_poly, scaler=scaler, resid_std=resid_std, z=1.96):
    Xs = scaler.transform(input_df)
    Xp = poly.transform(Xs)
    preds = model.predict(Xp)
    lower = preds - z*resid_std
    upper = preds + z*resid_std
    out = input_df.copy()
    out['predicted_harga'] = preds
    out['ci_lower'] = lower
    out['ci_upper'] = upper
    return out

new_samples = pd.DataFrame({
    'luas_tanah': [120, 300, 80, 450, 200],
    'luas_bangunan': [80, 250, 60, 350, 150],
    'jml_kamar': [2,4,1,5,3],
    'umur': [5,10,2,20,8],
    'jarak': [3,12,5,2,8]
})

preds_new = predict_property(new_samples)
print(preds_new)

# ================================================================
# 6. Kesimpulan
# ================================================================
# Model polynomial regression dengan derajat 2-3 + Ridge memberikan keseimbangan terbaik
# antara akurasi dan generalisasi. Lasso cocok untuk seleksi fitur.
# Rekomendasi: gunakan Ridge(alpha=1) dengan degree=2 atau 3 sebagai model final.
