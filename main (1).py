"""
Prediksi Durasi Pengiriman Supply Chain (DataCo) dengan Multiple Linear Regression.

Tugas Pengganti UTS Analitika Data
Anggota kelompok:
    - Aditya Surya Pratama   (I0324103)
    - Mohammad Arief Rahman  (I0324126)

Pembagian data : Training 64%, Validation 16%, Test 20% (di-split per Order Id).
Cara menjalankan:
    1. Letakkan DataCoSupplyChainDataset.csv di folder yang sama dengan main.py
    2. pip install -r requirements.txt
    3. python main.py
Semua tabel dan gambar hasil disimpan di folder ./output
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # hanya menyimpan gambar, tidak membuka jendela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Dengan drop="first" dan handle_unknown="ignore", sklearn memberi peringatan
# jika ada kategori yang tidak muncul di data latih. Perilakunya sudah benar
# (kategori tak dikenal dianggap sama dengan kategori acuan).
warnings.filterwarnings("ignore", message="Found unknown categories")


# ==========================================================================
# KONFIGURASI
# ==========================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = BASE_DIR / "output"

RANDOM_STATE = 42
TARGET = "Days for shipping (real)"
SCHEDULED = "Days for shipment (scheduled)"
MODE = "Shipping Mode"
GROUP = "Order Id"  # satu order bisa punya beberapa baris item

TEST_SIZE = 0.20  # porsi test dari seluruh data
VAL_SIZE = 0.20  # porsi validation dari data development (=16% total)
N_FOLDS = 5

FEATURE_SETS = {
    "A. Awal (14 fitur kode lama)": {
        "num": ["Order Item Quantity", "Order Item Discount",
                "Order Item Discount Rate", "Order Item Product Price",
                "Product Price", "Order Item Total", "Sales",
                "Latitude", "Longitude", SCHEDULED],
        "cat": [MODE, "Market", "Order Region", "Order Country"],
    },
    "B. Tanpa redundansi (model utama)": {
        "num": ["Order Item Quantity", "Order Item Discount Rate",
                "Order Item Product Price", "Latitude", "Longitude"],
        "cat": [MODE, "Order Region"],
    },
    "C. Shipping Mode saja": {
        "num": [],
        "cat": [MODE],
    },
}
FINAL_SET = "B. Tanpa redundansi (model utama)"

CORR_COLUMNS = [
    TARGET, SCHEDULED, "Order Item Quantity", "Order Item Discount",
    "Order Item Discount Rate", "Order Item Product Price",
    "Order Item Profit Ratio", "Sales", "Order Item Total",
    "Product Price", "Latitude", "Longitude",
]


# ==========================================================================
# FUNGSI UMUM
# ==========================================================================

def setup_environment():
    """Siapkan folder output dan tema grafik."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["savefig.dpi"] = 200


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def save_fig(filename):
    """Rapikan layout, simpan ke folder output, lalu tutup figure."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def regression_metrics(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "Dataset": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def print_metrics(metrics):
    print(f"\nEvaluasi {metrics['Dataset']}:")
    print("-" * 45)
    for key, value in metrics.items():
        if key != "Dataset":
            print(f"{key}: {value:.4f}")


# ==========================================================================
# 1. LOAD DATASET
# ==========================================================================

def load_data():
    section("1. LOAD DATASET")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}\n"
            "Letakkan DataCoSupplyChainDataset.csv di folder yang sama "
            "dengan main.py."
        )

    df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)

    print(f"Jumlah baris: {df.shape[0]:,}")
    print(f"Jumlah kolom: {df.shape[1]}")
    print(f"Ukuran memori: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Jumlah Order Id unik: {df[GROUP].nunique():,}")
    print("\nLima data teratas:")
    print(df.head().to_string(index=False))

    required = [TARGET, SCHEDULED, MODE, GROUP]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom wajib tidak ditemukan: {missing}")
    return df


# ==========================================================================
# 2. DATA CLEANSING & EDA
# ==========================================================================

def report_missing_and_duplicates(df):
    missing_report = pd.DataFrame({
        "Jumlah Missing": df.isnull().sum(),
        "Persentase Missing (%)": (df.isnull().mean() * 100).round(2),
    })
    missing_report = missing_report[missing_report["Jumlah Missing"] > 0]
    missing_report = missing_report.sort_values("Persentase Missing (%)",
                                                ascending=False)
    print("\nMissing values:")
    print(missing_report.to_string())
    missing_report.to_csv(OUTPUT_DIR / "missing_values.csv")

    print(f"\nJumlah baris duplikat: {int(df.duplicated().sum()):,}")


def report_descriptive_stats(df):
    df.describe(include="all").T.to_csv(OUTPUT_DIR / "statistik_deskriptif.csv")
    print("\nStatistik deskriptif (numerik):")
    print(df.describe().T.to_string())


def plot_target_distribution(df):
    values = pd.to_numeric(df[TARGET], errors="coerce").dropna()
    bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)

    plt.figure(figsize=(9, 5))
    sns.histplot(x=values, bins=bins)
    plt.title("Distribusi Durasi Pengiriman Aktual")
    plt.xlabel("Durasi Pengiriman (Hari)")
    plt.ylabel("Jumlah Observasi")
    save_fig("distribusi_target.png")


def plot_mode_distribution(df):
    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, y=MODE, order=df[MODE].value_counts().index)
    plt.title("Distribusi Moda Pengiriman")
    plt.xlabel("Jumlah Observasi")
    plt.ylabel("Moda Pengiriman")
    save_fig("distribusi_moda.png")


def report_duration_by_mode(df):
    summary = df.groupby(MODE)[TARGET].agg(
        ["count", "mean", "std", "min", "max"]).round(3)
    print("\nDurasi aktual per moda pengiriman:")
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / "durasi_per_moda.csv")

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x=MODE, y=TARGET,
                order=df.groupby(MODE)[TARGET].mean().sort_values().index)
    plt.title("Durasi Pengiriman Aktual per Moda Pengiriman")
    plt.xlabel("Moda Pengiriman")
    plt.ylabel("Durasi Aktual (Hari)")
    save_fig("boxplot_durasi_per_moda.png")


def report_actual_vs_scheduled_gap(df):
    gap = (pd.to_numeric(df[TARGET], errors="coerce")
           - pd.to_numeric(df[SCHEDULED], errors="coerce"))
    print("\nRingkasan selisih durasi aktual - jadwal:")
    print(gap.describe().to_string())

    counts = gap.value_counts().sort_index()
    plt.figure(figsize=(9, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Selisih Durasi Aktual dan Jadwal")
    plt.xlabel("Selisih Hari (Aktual - Jadwal)")
    plt.ylabel("Jumlah Observasi")
    save_fig("selisih_durasi.png")


def report_correlation(df):
    columns = [c for c in CORR_COLUMNS if c in df.columns]
    corr = df[columns].corr(numeric_only=True)

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap Variabel Numerik")
    save_fig("correlation_heatmap.png")

    target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs,
                                                        ascending=False)
    target_corr.to_csv(OUTPUT_DIR / "korelasi_target_eda.csv",
                       header=["Korelasi"])
    print("\nKorelasi numerik terhadap target:")
    print(target_corr.to_string())


def run_eda(df):
    section("2. DATA CLEANSING & EDA")
    report_missing_and_duplicates(df)
    report_descriptive_stats(df)
    plot_target_distribution(df)
    plot_mode_distribution(df)
    report_duration_by_mode(df)
    report_actual_vs_scheduled_gap(df)
    report_correlation(df)


# ==========================================================================
# 3. PEMERIKSAAN REDUNDANSI & PEMILIHAN FITUR
# ==========================================================================

def check_redundancy(df):
    """Cetak hasil pemeriksaan yang menjadi dasar pembuangan fitur."""
    section("3. PEMERIKSAAN REDUNDANSI & PEMILIHAN FITUR")

    checks = []

    if {"Product Price", "Order Item Product Price"} <= set(df.columns):
        share = (df["Product Price"] == df["Order Item Product Price"]).mean()
        checks.append(("Product Price == Order Item Product Price", share))

    if {"Sales", "Order Item Product Price", "Order Item Quantity"} <= set(df.columns):
        share = np.isclose(
            df["Sales"],
            df["Order Item Product Price"] * df["Order Item Quantity"],
        ).mean()
        checks.append(("Sales == Harga x Kuantitas", share))

    if {"Order Item Total", "Sales", "Order Item Discount"} <= set(df.columns):
        share = np.isclose(
            df["Order Item Total"], df["Sales"] - df["Order Item Discount"]
        ).mean()
        checks.append(("Order Item Total == Sales - Diskon", share))

    scheduled_per_mode = df.groupby(MODE)[SCHEDULED].nunique().max()
    checks.append((f"Jadwal ditentukan penuh oleh {MODE} "
                   "(maks nunique/moda = 1)",
                   float(scheduled_per_mode == 1)))

    if {"Market", "Order Region"} <= set(df.columns):
        region_in_market = df.groupby("Order Region")["Market"].nunique().max()
        checks.append(("Setiap Order Region hanya ada di satu Market "
                       "(maks nunique = 1)", float(region_in_market == 1)))

    target_per_order = df.groupby(GROUP)[TARGET].nunique().max()
    checks.append((f"Target konstan di dalam satu {GROUP} (maks nunique = 1)",
                   float(target_per_order == 1)))

    for label, value in checks:
        print(f"- {label}: {value * 100:.2f}%")

    print(
        "\nKesimpulan pemeriksaan:\n"
        "- Product Price, Sales, Order Item Total, Order Item Discount adalah\n"
        "  turunan dari harga, kuantitas, dan diskon -> dibuang.\n"
        "- Days for shipment (scheduled) adalah fungsi pasti dari Shipping Mode\n"
        "  (kombinasi linear dummy moda) -> dibuang, Shipping Mode dipertahankan.\n"
        "- Order Region bersarang di Market -> dummy keduanya kolinear sempurna;\n"
        "  dipakai Order Region saja. Order Country (164 level) dibuang agar\n"
        "  model tidak terlalu kompleks dan tidak ada kategori langka.\n"
        f"- Target konstan per {GROUP} -> split wajib per {GROUP}."
    )


def filter_feature_sets(df):
    """Buang fitur yang tidak ada di dataset (jaga-jaga)."""
    for spec in FEATURE_SETS.values():
        spec["num"] = [c for c in spec["num"] if c in df.columns]
        spec["cat"] = [c for c in spec["cat"] if c in df.columns]


def prepare_model_data(df):
    """Bentuk tabel pemodelan: semua kolom fitur + target + grup."""
    all_feature_cols = sorted({c for spec in FEATURE_SETS.values()
                               for c in spec["num"] + spec["cat"]})

    model_df = df[all_feature_cols + [TARGET, GROUP]].copy()
    model_df[TARGET] = pd.to_numeric(model_df[TARGET], errors="coerce")
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=[TARGET]).reset_index(drop=True)

    final_num = FEATURE_SETS[FINAL_SET]["num"]
    final_cat = FEATURE_SETS[FINAL_SET]["cat"]

    print(f"\nModel utama: {FINAL_SET}")
    print("Fitur numerik    :", final_num)
    print("Fitur kategorikal:", final_cat)
    print(f"Target           : {TARGET}")
    print(f"Jumlah observasi : {len(model_df):,}")

    return model_df, all_feature_cols, final_num, final_cat


# ==========================================================================
# 4. DATA SPLITTING PER ORDER ID
# ==========================================================================

@dataclass
class DataSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    g_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    g_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    g_test: pd.Series
    val_pos: np.ndarray  # posisi baris validation terhadap model_df
    test_pos: np.ndarray  # posisi baris test terhadap model_df


def group_split(groups, test_size, seed):
    """Kembalikan indeks posisi (train, test) dengan grup tidak beririsan."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=seed)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def split_data(X, y, groups):
    section("4. DATA SPLITTING (PER ORDER ID)")

    dev_pos, test_pos = group_split(groups, TEST_SIZE, RANDOM_STATE)
    X_dev, y_dev, g_dev = X.iloc[dev_pos], y.iloc[dev_pos], groups.iloc[dev_pos]

    train_pos, val_pos = group_split(g_dev, VAL_SIZE, RANDOM_STATE)

    split = DataSplit(
        X_train=X_dev.iloc[train_pos], y_train=y_dev.iloc[train_pos],
        g_train=g_dev.iloc[train_pos],
        X_val=X_dev.iloc[val_pos], y_val=y_dev.iloc[val_pos],
        g_val=g_dev.iloc[val_pos],
        X_test=X.iloc[test_pos], y_test=y.iloc[test_pos],
        g_test=groups.iloc[test_pos],
        val_pos=dev_pos[val_pos],  # posisi global terhadap model_df
        test_pos=test_pos,
    )

    # Pastikan tidak ada Order Id yang bocor antar subset
    assert set(split.g_train).isdisjoint(split.g_val), \
        "Order Id bocor antara train dan val"
    assert set(split.g_train).isdisjoint(split.g_test), \
        "Order Id bocor antara train dan test"
    assert set(split.g_val).isdisjoint(split.g_test), \
        "Order Id bocor antara val dan test"

    n_total = len(X)
    for label, X_part, g_part in [
        ("Training  ", split.X_train, split.g_train),
        ("Validation", split.X_val, split.g_val),
        ("Test      ", split.X_test, split.g_test),
    ]:
        print(f"{label}: {len(X_part):,} baris "
              f"({len(X_part) / n_total * 100:.2f}%) "
              f"| {g_part.nunique():,} order")
    print("Catatan: persentase baris mendekati 64/16/20 tetapi tidak persis, "
          "karena yang dibagi adalah order (satu order bisa berisi beberapa "
          "baris).")
    return split


# ==========================================================================
# 5. PREPROCESSING & PIPELINE MODEL
# ==========================================================================

def build_pipeline(numeric_cols, categorical_cols):
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), categorical_cols))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
        ("regressor", LinearRegression()),
    ])


def show_preprocessing(final_num, final_cat):
    section("5. PREPROCESSING")
    print("Fitur numerik (distandardisasi)        :", final_num)
    print("Fitur kategorikal (one-hot, drop first):", final_cat)


# ==========================================================================
# 6. BASELINE
# ==========================================================================

def evaluate_baselines(split, scheduled_all):
    section("6. BASELINE PEMBANDING")

    pred_mean = DummyRegressor(strategy="mean").fit(
        split.X_train, split.y_train).predict(split.X_val)
    pred_sched = scheduled_all.iloc[split.val_pos].to_numpy()

    baseline_df = pd.DataFrame([
        {"Baseline": "Rata-rata target training",
         "MAE": mean_absolute_error(split.y_val, pred_mean),
         "R2": r2_score(split.y_val, pred_mean)},
        {"Baseline": "Tebak = durasi terjadwal",
         "MAE": mean_absolute_error(split.y_val, pred_sched),
         "R2": r2_score(split.y_val, pred_sched)},
    ])
    baseline_df.to_csv(OUTPUT_DIR / "hasil_baseline.csv", index=False)
    print("Baseline pada validation set:")
    print(baseline_df.to_string(index=False))


# ==========================================================================
# 7. PERBANDINGAN SET FITUR
# ==========================================================================

def compare_feature_sets(split):
    section("7. PERBANDINGAN SET FITUR")

    cv = GroupKFold(n_splits=N_FOLDS)
    rows = []
    for name, spec in FEATURE_SETS.items():
        cols = spec["num"] + spec["cat"]
        pipe = build_pipeline(spec["num"], spec["cat"]).fit(
            split.X_train[cols], split.y_train)
        pred_val = pipe.predict(split.X_val[cols])

        cv_res = cross_validate(
            build_pipeline(spec["num"], spec["cat"]),
            split.X_train[cols], split.y_train, groups=split.g_train, cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"}, n_jobs=-1,
        )
        rows.append({
            "Set Fitur": name,
            "Jumlah Fitur": len(cols),
            "Val MAE": mean_absolute_error(split.y_val, pred_val),
            "Val R2": r2_score(split.y_val, pred_val),
            "CV MAE (mean)": -cv_res["test_mae"].mean(),
            "CV MAE (std)": cv_res["test_mae"].std(),
            "CV R2 (mean)": cv_res["test_r2"].mean(),
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(OUTPUT_DIR / "perbandingan_set_fitur.csv", index=False)
    print(comparison_df.round(4).to_string(index=False))
    print(
        "\nCara membaca: jika Val/CV hampir sama antar set, fitur tambahan "
        "tidak\nmenambah daya prediksi, jadi model yang lebih ringkas lebih "
        "dapat dipercaya."
    )


# ==========================================================================
# 8. MODEL UTAMA: TRAINING, VALIDATION, CROSS-VALIDATION
# ==========================================================================

def train_main_model(split, final_num, final_cat):
    section("8. MODEL UTAMA")

    cols = final_num + final_cat
    model = build_pipeline(final_num, final_cat)
    model.fit(split.X_train[cols], split.y_train)
    print("Training model selesai.")

    val_pred = model.predict(split.X_val[cols])
    val_metrics = regression_metrics(split.y_val, val_pred, "Validation")
    print_metrics(val_metrics)

    cv_res = cross_validate(
        build_pipeline(final_num, final_cat),
        split.X_train[cols], split.y_train, groups=split.g_train,
        cv=GroupKFold(n_splits=N_FOLDS),
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    cv_mae = -cv_res["test_score"]
    print(f"\nCross-validation {N_FOLDS}-fold (GroupKFold per Order Id)")
    print("MAE setiap fold:", np.round(cv_mae, 4))
    print(f"Rata-rata MAE CV: {cv_mae.mean():.4f}")
    print(f"Standar deviasi MAE CV: {cv_mae.std():.4f}")

    return model, val_metrics, cv_mae


# ==========================================================================
# 9. REGRESI LINEAR BERGANDA: TABEL OLS & UJI SIGNIFIKANSI
# ==========================================================================

def build_design(frame, numeric_cols, categorical_cols, train_columns=None):
    """Matriks desain (satuan asli, dummy drop-first) + kolom konstanta."""
    design = pd.get_dummies(frame[numeric_cols + categorical_cols],
                            columns=categorical_cols, drop_first=True,
                            dtype=float).astype(float)
    if train_columns is not None:
        design = design.reindex(columns=train_columns, fill_value=0.0)
    design = design.fillna(design.median())
    design.insert(0, "const", 1.0)
    return design


def ols_inference(design, y, groups):
    """OLS dengan SE klasik dan SE cluster (CR1) per grup."""
    X = design.to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ yv
    resid = yv - X @ beta
    rank = np.linalg.matrix_rank(X)

    # SE klasik
    sigma2 = resid @ resid / (n - rank)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t_vals = beta / se
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df=n - rank)

    # SE cluster per Order Id (baris dalam satu order tidak independen)
    scores = pd.DataFrame(X * resid[:, None]).groupby(
        np.asarray(groups)).sum().to_numpy()
    n_groups = scores.shape[0]
    meat = scores.T @ scores
    correction = (n_groups / (n_groups - 1)) * ((n - 1) / (n - rank))
    cov_cluster = correction * xtx_inv @ meat @ xtx_inv
    se_c = np.sqrt(np.diag(cov_cluster))
    t_c = beta / se_c
    p_c = 2 * stats.t.sf(np.abs(t_c), df=n_groups - 1)
    t_crit = stats.t.ppf(0.975, df=n_groups - 1)

    table = pd.DataFrame({
        "Koefisien": beta,
        "SE_klasik": se, "t_klasik": t_vals, "p_klasik": p_vals,
        "SE_cluster": se_c, "t_cluster": t_c, "p_cluster": p_c,
        "CI95_bawah": beta - t_crit * se_c,
        "CI95_atas": beta + t_crit * se_c,
    }, index=design.columns)

    ss_res = resid @ resid
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    k_pred = rank - 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - rank)
    f_stat = ((ss_tot - ss_res) / k_pred) / (ss_res / (n - rank))
    f_p = stats.f.sf(f_stat, k_pred, n - rank)
    summary_stats = {
        "n": n, "n_order": n_groups, "jumlah_prediktor": k_pred,
        "R2": r2, "Adj_R2": adj_r2, "F": f_stat, "p_F": f_p,
        "rank_matriks": rank, "jumlah_kolom": k,
    }
    return table, summary_stats, resid, X @ beta


def run_ols(split, final_num, final_cat):
    section("9. TABEL OLS & UJI SIGNIFIKANSI")

    design = build_design(split.X_train, final_num, final_cat)
    table, ols_stats, resid, fitted = ols_inference(
        design, split.y_train, split.g_train)

    table.to_csv(OUTPUT_DIR / "tabel_ols.csv")
    pd.Series(ols_stats).to_csv(OUTPUT_DIR / "ringkasan_ols.csv",
                                header=["Nilai"])

    print("Ringkasan model (data training, satuan variabel asli):")
    for key, value in ols_stats.items():
        print(f"  {key}: {value:.6g}" if isinstance(value, float)
              else f"  {key}: {value}")
    if ols_stats["rank_matriks"] < ols_stats["jumlah_kolom"]:
        print("  PERINGATAN: matriks desain tidak full-rank "
              "(masih ada kolinearitas sempurna).")

    print("\nKoefisien (p_cluster dipakai untuk kesimpulan karena baris dalam "
          "satu order saling berkorelasi):")
    show = table[["Koefisien", "SE_cluster", "p_cluster",
                  "CI95_bawah", "CI95_atas"]].copy()
    show["Signifikan (5%)"] = np.where(show["p_cluster"] < 0.05, "ya", "tidak")
    print(show.round(5).to_string())

    return design, ols_stats, resid, fitted


# ==========================================================================
# 10. MULTIKOLINEARITAS (VIF)
# ==========================================================================

def vif_table(frame):
    """VIF_j = 1 / (1 - R2_j), regresi kolom j terhadap kolom lain."""
    data = frame.to_numpy(dtype=float)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    rows = []
    for j, col in enumerate(frame.columns):
        others = np.delete(data, j, axis=1)
        design = np.column_stack([np.ones(len(data)), others])
        coef, *_ = np.linalg.lstsq(design, data[:, j], rcond=None)
        resid = data[:, j] - design @ coef
        r2 = 1 - (resid @ resid) / (data[:, j] @ data[:, j])
        vif = np.inf if r2 > 1 - 1e-10 else 1.0 / (1.0 - r2)
        rows.append({"Variabel": col, "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def run_vif(X_train, final_num):
    section("10. MULTIKOLINEARITAS (VIF)")

    numeric_before = FEATURE_SETS["A. Awal (14 fitur kode lama)"]["num"]
    vif_before = vif_table(
        X_train[numeric_before].fillna(X_train[numeric_before].median()))
    vif_before.to_csv(OUTPUT_DIR / "vif_sebelum.csv", index=False)
    print("VIF fitur numerik SEBELUM seleksi (set A):")
    print(vif_before.round(2).to_string(index=False))

    vif_after = vif_table(
        X_train[final_num].fillna(X_train[final_num].median()))
    vif_after.to_csv(OUTPUT_DIR / "vif_sesudah.csv", index=False)
    print("\nVIF fitur numerik SESUDAH seleksi (model utama):")
    print(vif_after.round(2).to_string(index=False))
    print("\nPedoman awal: VIF > 5 perlu diperiksa; VIF > 10 menunjukkan "
          "indikasi multikolinearitas kuat. Ini bukan batas mutlak.")


# ==========================================================================
# 11. DIAGNOSTIK ASUMSI REGRESI
# ==========================================================================

def breusch_pagan(design_arr, resid):
    """Uji Breusch-Pagan (versi LM): regresikan residual^2 pada prediktor."""
    aux_target = resid ** 2
    aux_coef, *_ = np.linalg.lstsq(design_arr, aux_target, rcond=None)
    aux_resid = aux_target - design_arr @ aux_coef
    aux_r2 = 1 - (aux_resid @ aux_resid) / (
        ((aux_target - aux_target.mean()) ** 2).sum())
    lm_stat = len(aux_target) * aux_r2
    df_bp = np.linalg.matrix_rank(design_arr) - 1
    return lm_stat, df_bp, stats.chi2.sf(lm_stat, df_bp)


def run_diagnostics(design, resid, fitted, X_train):
    section("11. DIAGNOSTIK ASUMSI REGRESI")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(resid), size=min(20000, len(resid)), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].scatter(fitted[idx], resid[idx], alpha=0.2, s=8)
    axes[0].axhline(0, linestyle="--", color="red")
    axes[0].set_title("Residual vs Nilai Prediksi")
    axes[0].set_xlabel("Nilai prediksi (hari)")
    axes[0].set_ylabel("Residual")

    stats.probplot(resid[idx], dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot Residual")

    sns.histplot(resid, bins=40, kde=True, ax=axes[2])
    axes[2].set_title("Distribusi Residual (Training)")
    axes[2].set_xlabel("Residual")
    save_fig("diagnostik_residual.png")

    bp_stat, bp_df, bp_p = breusch_pagan(design.to_numpy(dtype=float), resid)
    print(f"Breusch-Pagan (homoskedastisitas): LM={bp_stat:.2f}, "
          f"df={bp_df}, p={bp_p:.3g}")
    print(f"Skewness residual: {stats.skew(resid):.3f} | "
          f"Excess kurtosis: {stats.kurtosis(resid):.3f}")
    print("Rata-rata residual:", round(float(resid.mean()), 6))

    resid_by_mode = pd.DataFrame({
        "Moda": X_train[MODE].to_numpy(), "Residual": resid,
    }).groupby("Moda")["Residual"].agg(["mean", "std", "count"]).round(4)
    print("\nResidual per moda pengiriman (training):")
    print(resid_by_mode.to_string())

    print(
        "\nCatatan interpretasi:\n"
        "- Target berupa bilangan bulat 0-6 hari, sehingga residual tidak akan\n"
        "  berdistribusi normal sempurna dan variansnya berbeda antar moda.\n"
        "- Dengan n besar, uji formal hampir selalu menolak H0; nilai praktisnya\n"
        "  dilihat dari plot dan ukuran efek. Karena itu inferensi memakai SE\n"
        "  cluster per Order Id dan keterbatasan ini dibahas di laporan."
    )


# ==========================================================================
# 12. PERMUTATION FEATURE IMPORTANCE
# ==========================================================================

def run_permutation_importance(model, split, cols_final):
    section("12. PERMUTATION FEATURE IMPORTANCE")

    result = permutation_importance(
        model, split.X_val[cols_final], split.y_val,
        scoring="neg_mean_absolute_error", n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "Variabel": cols_final,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "peringkat_kepentingan_variabel.csv",
                         index=False)
    print(importance_df.round(5).to_string(index=False))

    plot_df = importance_df.sort_values("Importance_Mean", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["Variabel"], plot_df["Importance_Mean"],
             xerr=plot_df["Importance_Std"], capsize=3)
    plt.axvline(0, linestyle="--", color="black")
    plt.title("Permutation Feature Importance pada Validation Set")
    plt.xlabel("Kenaikan MAE (hari) saat variabel diacak")
    plt.ylabel("Variabel")
    save_fig("peringkat_kepentingan_variabel.png")

    top = importance_df.iloc[0]
    print(f"\nVariabel terpenting: {top['Variabel']} "
          f"(kenaikan MAE {top['Importance_Mean']:.4f} hari saat diacak)")
    print("Nilai mendekati 0 berarti variabel hampir tidak membantu prediksi.")


# ==========================================================================
# 13. FINAL TRAINING & EVALUASI TEST SET
# ==========================================================================

def final_evaluation(split, scheduled_all, final_num, final_cat, val_metrics):
    section("13. FINAL TRAINING & TESTING")

    cols = final_num + final_cat
    X_final = pd.concat([split.X_train, split.X_val], axis=0)[cols]
    y_final = pd.concat([split.y_train, split.y_val], axis=0)
    final_model = build_pipeline(final_num, final_cat).fit(X_final, y_final)

    y_test = split.y_test
    test_pred = final_model.predict(split.X_test[cols])
    test_metrics = regression_metrics(y_test, test_pred, "Test")
    print_metrics(test_metrics)

    # Pembanding pada test set
    base_pred = np.full(len(y_test), y_final.mean())
    sched_pred = scheduled_all.iloc[split.test_pos].to_numpy()
    print("\nPembanding pada test set:")
    print(f"  Baseline rata-rata : MAE={mean_absolute_error(y_test, base_pred):.4f}"
          f" | R2={r2_score(y_test, base_pred):.4f}")
    print(f"  Tebak = jadwal     : MAE={mean_absolute_error(y_test, sched_pred):.4f}"
          f" | R2={r2_score(y_test, sched_pred):.4f}")

    # Galat per moda: menunjukkan di mana model kuat dan di mana ada noise
    report = pd.DataFrame({
        "Moda": split.X_test[MODE].to_numpy(),
        "Aktual": y_test.to_numpy(),
        "Prediksi": test_pred,
    })
    report["Abs_Error"] = (report["Aktual"] - report["Prediksi"]).abs()
    error_by_mode = report.groupby("Moda").agg(
        n=("Aktual", "size"), MAE=("Abs_Error", "mean"),
        Rata2_Aktual=("Aktual", "mean"), Rata2_Prediksi=("Prediksi", "mean"),
    ).round(4)
    error_by_mode.to_csv(OUTPUT_DIR / "galat_per_moda.csv")
    print("\nGalat test set per moda pengiriman:")
    print(error_by_mode.to_string())

    test_resid = y_test.to_numpy() - test_pred
    plot_test_results(y_test, test_pred, test_resid)

    pd.DataFrame([val_metrics, test_metrics]).to_csv(
        OUTPUT_DIR / "hasil_evaluasi.csv", index=False)
    pd.DataFrame({
        "Durasi_Aktual": y_test.to_numpy(),
        "Durasi_Prediksi": test_pred,
        "Residual": test_resid,
    }).to_csv(OUTPUT_DIR / "prediksi_test.csv", index=False)

    return test_metrics


def plot_test_results(y_test, test_pred, test_resid):
    rng = np.random.default_rng(RANDOM_STATE)
    jitter = rng.uniform(-0.15, 0.15, size=len(y_test))

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test.to_numpy() + jitter, test_pred, alpha=0.15, s=8)
    lo = min(float(y_test.min()), float(test_pred.min()))
    hi = max(float(y_test.max()), float(test_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="red")
    plt.title("Aktual vs Prediksi Durasi Pengiriman (Test Set)")
    plt.xlabel("Durasi Aktual (Hari, diberi jitter kecil)")
    plt.ylabel("Durasi Prediksi (Hari)")
    save_fig("aktual_vs_prediksi.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(test_resid, bins=30, kde=True)
    plt.title("Distribusi Residual pada Test Set")
    plt.xlabel("Residual (Aktual - Prediksi)")
    plt.ylabel("Frekuensi")
    save_fig("distribusi_residual.png")


# ==========================================================================
# 14. RINGKASAN AKHIR
# ==========================================================================

def save_summary(model_df, split, cols_final, val_metrics, test_metrics,
                 cv_mae, ols_stats):
    section("14. RINGKASAN AKHIR")

    summary = pd.Series({
        "Jumlah observasi pemodelan": len(model_df),
        "Jumlah order unik": model_df[GROUP].nunique(),
        "Jumlah fitur (model utama)": len(cols_final),
        "Training set (baris)": len(split.X_train),
        "Validation set (baris)": len(split.X_val),
        "Test set (baris)": len(split.X_test),
        "Validation MAE": val_metrics["MAE"],
        "Rata-rata MAE cross-validation": cv_mae.mean(),
        "Test MAE": test_metrics["MAE"],
        "Test RMSE": test_metrics["RMSE"],
        "Test R2": test_metrics["R2"],
        "Adj R2 (training, OLS)": ols_stats["Adj_R2"],
    })
    summary.to_csv(OUTPUT_DIR / "ringkasan_akhir.csv", header=["Nilai"])
    print(summary.to_string())
    print("\nSeluruh output disimpan pada:")
    print(OUTPUT_DIR.resolve())
    print("\nProgram selesai.")


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    setup_environment()

    df = load_data()
    run_eda(df)

    check_redundancy(df)
    filter_feature_sets(df)
    model_df, all_feature_cols, final_num, final_cat = prepare_model_data(df)
    cols_final = final_num + final_cat

    X_all = model_df[all_feature_cols]
    y_all = model_df[TARGET]
    groups_all = model_df[GROUP]
    scheduled_all = pd.to_numeric(model_df[SCHEDULED], errors="coerce")

    split = split_data(X_all, y_all, groups_all)

    show_preprocessing(final_num, final_cat)
    evaluate_baselines(split, scheduled_all)
    compare_feature_sets(split)

    model, val_metrics, cv_mae = train_main_model(split, final_num, final_cat)
    design, ols_stats, resid, fitted = run_ols(split, final_num, final_cat)
    run_vif(split.X_train, final_num)
    run_diagnostics(design, resid, fitted, split.X_train)
    run_permutation_importance(model, split, cols_final)

    test_metrics = final_evaluation(split, scheduled_all, final_num,
                                    final_cat, val_metrics)
    save_summary(model_df, split, cols_final, val_metrics, test_metrics,
                 cv_mae, ols_stats)


if __name__ == "__main__":
    main()
