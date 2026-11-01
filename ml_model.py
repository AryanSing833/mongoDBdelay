'''ml_model.py
Train on ecommerce_logistics_delivery.csv and test on delivery_data.csv.

The two files do not share the same schema. This script:
  1. Prints a feature comparison.
  2. Maps overlapping / convertible columns onto a shared feature space.
  3. Trains a Random Forest on the ecommerce dataset.
  4. Evaluates on delivery_data.csv (held-out test set).
  5. Also reports an in-domain 80/20 split on ecommerce (full feature set).
'''

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "ecommerce_logistics_delivery.csv"
TEST_CSV = BASE_DIR / "delivery_data.csv"

# Weather categories in delivery_data.csv → minutes, matching the
# ecommerce Weather_Delay_Minutes scale (mostly 0–10, mean ~3.8).
WEATHER_TO_DELAY_MINUTES = {
    "Sunny": 0.0,
    "Windy": 2.0,
    "Fog": 4.0,
    "Rainy": 6.0,
    "Snow": 10.0,
}

# traffic_index in delivery_data.csv is ~0.5–2.5 (unitless).
# Map onto Traffic_Delay_Minutes (~0–60 typical, mean ~22).
TRAFFIC_INDEX_TO_MINUTES = 15.0

SHARED_FEATURES = [
    "distance_km",
    "package_weight_kg",
    "traffic_delay_minutes",
    "weather_delay_minutes",
]


def compare_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print column overlap and how each test column is used."""
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    print("\n" + "=" * 64)
    print("  DATASET COMPARISON")
    print("=" * 64)
    print(f"  Train : {TRAIN_CSV.name}  ({len(train_df):,} rows, {len(train_df.columns)} cols)")
    print(f"  Test  : {TEST_CSV.name}  ({len(test_df):,} rows, {len(test_df.columns)} cols)")

    exact = sorted(c for c in test_cols if c in train_cols)
    print(f"\n  Exact same column names : {exact if exact else '(none)'}")

    print("\n  Test column              Train counterpart                 Status")
    print("  " + "-" * 70)
    mapping_rows = [
        ("order_id", "Order_ID", "IDs only - not a feature"),
        ("driver_id", "Delivery_Partner", "different IDs - dropped"),
        ("distance_km", "Distance_KM", "MATCH - renamed"),
        ("package_weight_kg", "Package_Weight_KG", "MATCH - renamed (scale differs)"),
        ("traffic_index", "Traffic_Delay_Minutes", "MAPPED - index * 15 -> minutes"),
        ("weather_condition", "Weather_Delay_Minutes", "MAPPED - category -> minutes"),
        ("delivery_status", "Delivery_Status", "different enums - not used as feature"),
        ("is_delayed", "On_Time_Delivery_Flag", "MATCH - inverted flag is the target"),
    ]
    for test_col, train_col, status in mapping_rows:
        print(f"  {test_col:<24} {train_col:<32} {status}")

    unused_train = sorted(
        train_cols
        - {
            "Order_ID",
            "Distance_KM",
            "Package_Weight_KG",
            "Traffic_Delay_Minutes",
            "Weather_Delay_Minutes",
            "On_Time_Delivery_Flag",
            "Delivery_Status",
            "Delivery_Partner",
        }
    )
    print("\n  Train-only columns (cannot be used on delivery_data.csv):")
    print("   ", ", ".join(unused_train))

    print("\n  Shared model features:")
    for name in SHARED_FEATURES:
        print(f"    - {name}")
    print("  Target: is_delayed  (0 = on time, 1 = delayed)")
    print("=" * 64)


def prepare_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Align ecommerce columns to the shared feature names."""
    out = pd.DataFrame({
        "distance_km": df["Distance_KM"].astype(float),
        "package_weight_kg": df["Package_Weight_KG"].astype(float),
        "traffic_delay_minutes": df["Traffic_Delay_Minutes"].astype(float),
        "weather_delay_minutes": df["Weather_Delay_Minutes"].astype(float),
        # 1 = delayed when the order was NOT on time
        "is_delayed": (df["On_Time_Delivery_Flag"].astype(int) == 0).astype(int),
    })
    return out[SHARED_FEATURES], out["is_delayed"]


def prepare_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Map delivery_data.csv onto the same shared features as training."""
    weather_delay = (
        df["weather_condition"]
        .map(WEATHER_TO_DELAY_MINUTES)
        .fillna(WEATHER_TO_DELAY_MINUTES["Sunny"])
        .astype(float)
    )
    out = pd.DataFrame({
        "distance_km": df["distance_km"].astype(float),
        "package_weight_kg": df["package_weight_kg"].astype(float),
        "traffic_delay_minutes": df["traffic_index"].astype(float) * TRAFFIC_INDEX_TO_MINUTES,
        "weather_delay_minutes": weather_delay,
        "is_delayed": df["is_delayed"].astype(int),
    })
    return out[SHARED_FEATURES], out["is_delayed"]


def print_metrics(y_true, preds, title: str, feature_cols: List[str], model=None) -> None:
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print(f"  Accuracy    : {accuracy_score(y_true, preds):.4f}")
    print(f"  Precision   : {precision_score(y_true, preds, zero_division=0):.4f}")
    print(f"  Recall      : {recall_score(y_true, preds, zero_division=0):.4f}")
    print(f"  F1-Score    : {f1_score(y_true, preds, zero_division=0):.4f}")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(y_true, preds, target_names=["On-Time", "Delayed"], zero_division=0))

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    print("Confusion Matrix:")
    print("              Predicted")
    print("              On-Time  Delayed")
    print(f"Actual On-Time   {cm[0][0]:>5}   {cm[0][1]:>5}")
    print(f"Actual Delayed   {cm[1][0]:>5}   {cm[1][1]:>5}")

    if model is not None and hasattr(model, "feature_importances_"):
        print("\nFeature Importances:")
        importances = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, imp in importances:
            bar = "#" * int(imp * 80)
            print(f"  {name:<28} {imp:.4f} {bar}")


def train_shared_and_test_delivery(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> None:
    X_train, y_train = prepare_train(train_raw)
    X_test, y_test = prepare_test(test_raw)

    missing = [c for c in SHARED_FEATURES if c not in X_train.columns or c not in X_test.columns]
    if missing:
        raise ValueError(f"Aligned features missing after mapping: {missing}")

    print("\n" + "=" * 64)
    print("  CROSS-DATASET RUN  (train=ecommerce, test=delivery_data.csv)")
    print("=" * 64)
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")
    print(f"  Train classes    : On-Time={(y_train == 0).sum()} | Delayed={(y_train == 1).sum()}")
    print(f"  Test classes     : On-Time={(y_test == 0).sum()} | Delayed={(y_test == 1).sum()}")
    print("\n  Feature check (train vs test means):")
    for col in SHARED_FEATURES:
        print(
            f"    {col:<24} train_mean={X_train[col].mean():8.2f}   "
            f"test_mean={X_test[col].mean():8.2f}"
        )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print("\nTraining Random Forest on shared features...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print_metrics(
        y_test,
        preds,
        "EVALUATION ON delivery_data.csv (TEST)",
        SHARED_FEATURES,
        model,
    )


def train_from_mongodb(
    db_name: str | None = None,
    collection_name: str = "orders",
) -> None:
    """Train on the ecommerce rows ingested into MongoDB (new dataset)."""
    client = MongoClient(
        os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017"),
        serverSelectionTimeoutMS=5000,
    )
    database_name = db_name or os.environ.get("MONGO_DB_NAME", "delivery_fleet")
    docs = list(client[database_name][collection_name].find({}, {"_id": 0}))
    if not docs:
        raise ValueError("No data in MongoDB. Run database.py first.")

    df = pd.DataFrame(docs)
    print("\n" + "=" * 64)
    print("  MONGODB RUN  (ecommerce_logistics_delivery ingested into orders)")
    print("=" * 64)
    print(f"  Documents loaded : {len(df)}")

    feature_cols = [
        "distance_km",
        "traffic_delay_minutes",
        "weather_delay_minutes",
        "package_weight_kg",
        "warehouse_processing_h",
        "courier_rating",
        "courier_experience_yrs",
        "warehouse_capacity_pct",
        "weekend_flag",
        "holiday_flag",
        "promotion_flag",
        "order_hour",
        "day_of_week",
        "month",
    ]
    missing = [c for c in feature_cols + ["is_delayed", "shipping_method"] if c not in df.columns]
    if missing:
        raise ValueError(f"MongoDB orders missing expected fields: {missing}")

    le = LabelEncoder()
    df["shipping_method_enc"] = le.fit_transform(df["shipping_method"].astype(str))
    feature_cols = feature_cols + ["shipping_method_enc"]

    X = df[feature_cols]
    y = df["is_delayed"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training samples : {len(X_tr)}")
    print(f"  Testing samples  : {len(X_te)}")
    print(f"  Train classes    : On-Time={(y_tr == 0).sum()} | Delayed={(y_tr == 1).sum()}")

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print("\nTraining Random Forest on MongoDB ecommerce features...")
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print_metrics(
        y_te,
        preds,
        "EVALUATION ON MongoDB ecommerce hold-out (is_delayed)",
        feature_cols,
        model,
    )


# IDs and post-delivery outcomes — not available at prediction time.
ECOMMERCE_DROP = [
    "Order_ID",
    "Order_Date",
    "Delivery_Status",
    "Delivery_Delay_Hours",
    "Estimated_Delivery_Hours",
    "Promised_Delivery_Hours",
    "Cancellation_Flag",
    "Return_Flag",
    "On_Time_Delivery_Flag",
    "Customer_Satisfaction_Score",
    "Shipping_Cost",
    "Logistics_Cost",
]


def encode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = LabelEncoder().fit_transform(out[col].astype(str))
    return out


def select_top_features(
    names: List[str],
    importances,
    min_features: int = 8,
    cumulative: float = 0.85,
) -> List[str]:
    ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
    selected: List[str] = []
    running = 0.0
    for name, imp in ranked:
        selected.append(name)
        running += float(imp)
        if len(selected) >= min_features and running >= cumulative:
            break
    return selected


def train_ecommerce_with_importance(train_raw: pd.DataFrame) -> None:
    """Train on ecommerce_logistics_delivery.csv using feature importance.

    Stage 1: fit Random Forest on all usable features and rank them.
    Stage 2: retrain on the features that explain ~85% of importance.
    """
    target = "On_Time_Delivery_Flag"
    X = train_raw.drop(columns=[c for c in ECOMMERCE_DROP if c in train_raw.columns])
    y = (train_raw[target].astype(int) == 0).astype(int)
    X = encode_object_columns(X)

    feature_cols = list(X.columns)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n" + "=" * 64)
    print("  ECOMMERCE CSV  (feature importance + retrain)")
    print("=" * 64)
    print(f"  File             : {TRAIN_CSV.name}")
    print(f"  Rows             : {len(train_raw):,}")
    print(f"  Training samples : {len(X_tr)}")
    print(f"  Testing samples  : {len(X_te)}")
    print(f"  Candidate feats  : {len(feature_cols)}")
    print(f"  Target           : is_delayed  (On_Time_Delivery_Flag inverted)")
    print(f"  Train classes    : On-Time={(y_tr == 0).sum()} | Delayed={(y_tr == 1).sum()}")

    print("\nStage 1 — ranking all ecommerce features...")
    rank_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rank_model.fit(X_tr, y_tr)
    print_metrics(
        y_te,
        rank_model.predict(X_te),
        "STAGE 1 — all ecommerce features",
        feature_cols,
        rank_model,
    )

    selected = select_top_features(feature_cols, rank_model.feature_importances_)
    print("\nSelected features (cumulative importance >= 85%, min 8):")
    ranked = sorted(
        zip(feature_cols, rank_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    running = 0.0
    for name, imp in ranked:
        running += float(imp)
        mark = "KEEP" if name in selected else "drop"
        print(f"  [{mark}] {name:<32} {imp:.4f}  cum={running:.3f}")

    print(f"\nStage 2 — retraining on {len(selected)} important features...")
    final = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final.fit(X_tr[selected], y_tr)
    print_metrics(
        y_te,
        final.predict(X_te[selected]),
        "STAGE 2 — important-feature model (TEST)",
        selected,
        final,
    )


if __name__ == "__main__":
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_CSV}")

    train_raw = pd.read_csv(TRAIN_CSV)
    train_ecommerce_with_importance(train_raw)
