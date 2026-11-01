'''app.py
Flask API + dashboard for the ecommerce logistics MongoDB dataset.
'''

from datetime import datetime
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

CSV_PATH = BASE_DIR / "ecommerce_logistics_delivery.csv"
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "delivery_fleet")

# Map CSV columns onto the dashboard / API field names.
CSV_FEATURE_MAP = {
    "Distance_KM": "distance_km",
    "Traffic_Delay_Minutes": "traffic_delay_minutes",
    "Weather_Delay_Minutes": "weather_delay_minutes",
    "Package_Weight_KG": "package_weight_kg",
    "Warehouse_Processing_Hours": "warehouse_processing_h",
    "Courier_Rating": "courier_rating",
    "Courier_Experience_Years": "courier_experience_yrs",
    "Warehouse_Capacity_Utilization": "warehouse_capacity_pct",
    "Weekend_Flag": "weekend_flag",
    "Holiday_Flag": "holiday_flag",
    "Promotion_Flag": "promotion_flag",
    "Order_Hour": "order_hour",
    "Day_of_Week": "day_of_week",
    "Month": "month",
    "Shipping_Method": "shipping_method",
    "Estimated_Delivery_Hours": "estimated_delivery_h",
    "Promised_Delivery_Hours": "promised_delivery_h",
    "Order_Value": "order_value",
    "Order_Quantity": "order_quantity",
    "Package_Volume_CM3": "package_volume_cm3",
    "Inventory_Availability": "inventory_availability",
    "Shipping_Cost": "shipping_cost",
    "Logistics_Cost": "logistics_cost",
}

CSV_DROP = {
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
}

FEATURE_DEFAULTS = {
    "distance_km": 180.0,
    "traffic_delay_minutes": 20.0,
    "weather_delay_minutes": 3.0,
    "package_weight_kg": 1.5,
    "warehouse_processing_h": 4.0,
    "courier_rating": 4.2,
    "courier_experience_yrs": 4.0,
    "warehouse_capacity_pct": 75.0,
    "weekend_flag": 0,
    "holiday_flag": 0,
    "promotion_flag": 0,
    "order_hour": 12,
    "day_of_week": 3,
    "month": 6,
    "estimated_delivery_h": 80.0,
    "promised_delivery_h": 72,
    "order_value": 80.0,
    "order_quantity": 2,
    "package_volume_cm3": 1200.0,
    "inventory_availability": 90.0,
    "shipping_cost": 22.0,
    "logistics_cost": 35.0,
}

MODEL_STATE = {
    "model": None,
    "encoders": {},
    "feature_cols": [],
    "metrics": {},
    "ready": False,
    "error": None,
    "labeled_outcomes": 0,
}

PREDICTION_COLLECTION = "prediction_inputs"


def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
    return client[MONGO_DB_NAME]


def _prediction_input_from_document(document: dict) -> dict:
    row = dict(FEATURE_DEFAULTS)
    for key in FEATURE_DEFAULTS:
        if key in document and document[key] is not None:
            row[key] = document[key]
    row["shipping_method"] = document.get("shipping_method", "Standard")
    return row


def _load_labeled_prediction_inputs(db) -> tuple[pd.DataFrame, pd.Series]:
    documents = list(db[PREDICTION_COLLECTION].find(
        {"actual_is_delayed": {"$exists": True}},
        {"_id": 0, "actual_is_delayed": 1, **{key: 1 for key in FEATURE_DEFAULTS}},
    ))
    if not documents:
        return pd.DataFrame(), pd.Series(dtype=int)
    rows = [_prediction_input_from_document(document) for document in documents]
    labels = pd.Series(
        [int(document["actual_is_delayed"]) for document in documents],
        dtype=int,
    )
    return pd.DataFrame(rows), labels


def risk_from_status(status: str, traffic_min: float) -> str:
    if status == "Severely Delayed":
        return "Critical"
    if status == "Delayed":
        return "High"
    if status == "Cancelled":
        return "Medium"
    if traffic_min > 50:
        return "Medium"
    return "Low"


def pretty_driver(driver_id: str, name: str | None = None) -> str:
    if name:
        return name
    return driver_id.replace("_", " ").title().replace("In-House", "In-House")


def _select_top_features(names, importances, min_features=8, cumulative=0.85):
    ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
    selected = []
    running = 0.0
    for name, imp in ranked:
        selected.append(name)
        running += float(imp)
        if len(selected) >= min_features and running >= cumulative:
            break
    return selected


def train_model() -> None:
    try:
        if not CSV_PATH.exists():
            MODEL_STATE["error"] = f"Dataset not found: {CSV_PATH.name}"
            return
        raw = pd.read_csv(CSV_PATH)
        y = (raw["On_Time_Delivery_Flag"].astype(int) == 0).astype(int)
        X = raw.drop(columns=[c for c in CSV_DROP if c in raw.columns])
        rename = {src: dst for src, dst in CSV_FEATURE_MAP.items() if src in X.columns}
        X = X.rename(columns=rename)
        db = get_db()
        feedback_X, feedback_y = _load_labeled_prediction_inputs(db)
        if not feedback_X.empty:
            feedback_X = feedback_X.reindex(columns=X.columns, fill_value=0)
            X = pd.concat([X, feedback_X], ignore_index=True)
            y = pd.concat([y, feedback_y], ignore_index=True)

        encoders = {}
        for col in X.select_dtypes(include="object").columns:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))
            encoders[col] = encoder

        all_cols = list(X.columns)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

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
        selected = _select_top_features(all_cols, rank_model.feature_importances_)

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr[selected], y_tr)
        preds = model.predict(X_te[selected])
        ranked = sorted(
            zip(selected, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )

        MODEL_STATE["model"] = model
        MODEL_STATE["encoders"] = encoders
        MODEL_STATE["feature_cols"] = selected
        MODEL_STATE["metrics"] = {
            "accuracy": round(float(accuracy_score(y_te, preds)) * 100, 1),
            "precision": round(float(precision_score(y_te, preds, zero_division=0)) * 100, 1),
            "recall": round(float(recall_score(y_te, preds, zero_division=0)) * 100, 1),
            "f1": round(float(f1_score(y_te, preds, zero_division=0)) * 100, 1),
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "source": CSV_PATH.name,
            "features": selected,
            "labeled_outcomes": int(len(feedback_y)),
        }
        MODEL_STATE["ready"] = True
        MODEL_STATE["error"] = None
        MODEL_STATE["labeled_outcomes"] = int(len(feedback_y))
        print("Selected features:", selected)
        for name, imp in ranked:
            print(f"  {name:<28} {imp:.4f}")
    except Exception as exc:
        MODEL_STATE["error"] = str(exc)
        MODEL_STATE["ready"] = False


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/api/health")
def health():
    try:
        db = get_db()
        db.command("ping")
        orders = db["orders"].count_documents({})
        drivers = db["drivers"].count_documents({})
        return jsonify({
            "ok": True,
            "mongodb": True,
            "orders": orders,
            "drivers": drivers,
            "model_ready": MODEL_STATE["ready"],
            "model_error": MODEL_STATE["error"],
        })
    except Exception as exc:
        return jsonify({"ok": False, "mongodb": False, "error": str(exc)}), 503


@app.route("/api/stats")
def stats():
    db = get_db()
    orders = db["orders"]
    total = orders.count_documents({})
    if total == 0:
        return jsonify({"error": "No orders in MongoDB"}), 404

    status_counts = {
        d["_id"]: d["n"]
        for d in orders.aggregate([{"$group": {"_id": "$delivery_status", "n": {"$sum": 1}}}])
    }
    on_time = status_counts.get("On Time", 0)
    delayed = status_counts.get("Delayed", 0)
    severe = status_counts.get("Severely Delayed", 0)
    cancelled = status_counts.get("Cancelled", 0)

    avg_delay = list(orders.aggregate([
        {"$group": {"_id": None, "avg": {"$avg": "$delivery_delay_hours"}}}
    ]))
    avg_transit = round(float(avg_delay[0]["avg"]), 1) if avg_delay else 0

    volume = [0] * 7
    for d in orders.aggregate([
        {"$group": {"_id": "$day_of_week", "n": {"$sum": 1}}}
    ]):
        dow = int(d["_id"])
        if 0 <= dow <= 6:
            volume[dow] = d["n"]

    model_metrics = dict(MODEL_STATE["metrics"])
    model_metrics["prediction_inputs"] = db[PREDICTION_COLLECTION].count_documents({})
    model_metrics["labeled_outcomes"] = db[PREDICTION_COLLECTION].count_documents(
        {"actual_is_delayed": {"$exists": True}},
    )
    model_metrics["next_retraining_at"] = (
        ((model_metrics["labeled_outcomes"] // 100) + 1) * 100
    )

    return jsonify({
        "totalOrders": total,
        "onTime": on_time,
        "delayed": delayed,
        "severelyDelayed": severe,
        "cancelled": cancelled,
        "onTimeRate": round(100.0 * on_time / total, 1),
        "highRiskCount": severe,
        "avgTransitHours": avg_transit,
        "statusCounts": [on_time, delayed, severe, cancelled],
        "volumeByDow": volume,
        "model": model_metrics,
    })


@app.route("/api/orders")
def list_orders():
    db = get_db()
    limit = min(int(request.args.get("limit", 80)), 200)
    status = request.args.get("status")
    query = {}
    if status and status != "All":
        query["delivery_status"] = status

    drivers = {d["driver_id"]: d.get("name", d["driver_id"]) for d in db["drivers"].find({})}
    cursor = db["orders"].find(query, {"_id": 0, "checkpoint_logs": 0}).sort("order_id", 1).limit(limit)

    rows = []
    for o in cursor:
        status_val = o.get("delivery_status", "")
        traffic = float(o.get("traffic_delay_minutes") or 0)
        driver_id = o.get("driver_id", "")
        rows.append({
            "id": o.get("order_id"),
            "driver": pretty_driver(driver_id, drivers.get(driver_id)),
            "driver_id": driver_id,
            "dist": round(float(o.get("distance_km") or 0), 2),
            "trafficMin": round(traffic, 1),
            "weatherMin": round(float(o.get("weather_delay_minutes") or 0), 1),
            "region": o.get("customer_region", ""),
            "category": o.get("product_category", ""),
            "shipping": o.get("shipping_method", ""),
            "weight": round(float(o.get("package_weight_kg") or 0), 2),
            "delayHours": round(float(o.get("delivery_delay_hours") or 0), 2),
            "status": status_val,
            "risk": risk_from_status(status_val, traffic),
        })
    return jsonify({"orders": rows, "count": len(rows)})


@app.route("/api/aggregations")
def aggregations():
    db = get_db()
    orders = db["orders"]

    pipeline_a = list(orders.aggregate([
        {"$group": {
            "_id": "$delivery_status",
            "order_count": {"$sum": 1},
            "avg_distance_km": {"$avg": "$distance_km"},
            "avg_delay_hours": {"$avg": "$delivery_delay_hours"},
            "avg_satisfaction": {"$avg": "$customer_satisfaction"},
            "total_logistics_cost": {"$sum": "$logistics_cost"},
        }},
        {"$sort": {"avg_delay_hours": -1}},
    ]))

    status_order = ["On Time", "Delayed", "Severely Delayed", "Cancelled"]
    by_status = {d["_id"]: d for d in pipeline_a}
    avg_distance = [round(float(by_status.get(s, {}).get("avg_distance_km") or 0), 2) for s in status_order]
    avg_delay = [round(float(by_status.get(s, {}).get("avg_delay_hours") or 0), 2) for s in status_order]

    shipping = list(orders.aggregate([
        {"$group": {
            "_id": "$shipping_method",
            "total_orders": {"$sum": 1},
            "on_time_count": {"$sum": {"$cond": [{"$eq": ["$is_delayed", 0]}, 1, 0]}},
        }},
    ]))
    ship_order = ["Economy", "Standard", "Express", "Same Day"]
    ship_map = {d["_id"]: d for d in shipping}
    on_time_rates = []
    for method in ship_order:
        d = ship_map.get(method, {"total_orders": 1, "on_time_count": 0})
        total = d["total_orders"] or 1
        on_time_rates.append(round(100.0 * d["on_time_count"] / total, 1))

    drivers_lookup = {d["driver_id"]: d.get("name", d["driver_id"]) for d in db["drivers"].find({})}
    driver_perf = list(orders.aggregate([
        {"$group": {
            "_id": "$driver_id",
            "total_orders": {"$sum": 1},
            "delayed_orders": {"$sum": "$is_delayed"},
        }},
        {"$project": {
            "driver_id": "$_id",
            "total_orders": 1,
            "delayed_orders": 1,
            "delay_rate": {"$divide": ["$delayed_orders", "$total_orders"]},
            "_id": 0,
        }},
        {"$sort": {"delay_rate": -1}},
    ]))
    for d in driver_perf:
        d["name"] = pretty_driver(d["driver_id"], drivers_lookup.get(d["driver_id"]))
        d["delay_rate"] = round(float(d["delay_rate"]), 3)

    return jsonify({
        "pipelineA": [
            {
                "status": d["_id"],
                "order_count": d["order_count"],
                "avg_distance_km": round(float(d["avg_distance_km"]), 2),
                "avg_delay_hours": round(float(d["avg_delay_hours"]), 2),
                "avg_satisfaction": round(float(d["avg_satisfaction"]), 2),
                "total_logistics_cost": round(float(d["total_logistics_cost"]), 2),
            }
            for d in pipeline_a
        ],
        "avgDistance": avg_distance,
        "avgDelay": avg_delay,
        "shippingMethods": ship_order,
        "onTimeRates": on_time_rates,
        "drivers": driver_perf,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if not MODEL_STATE["ready"]:
        return jsonify({"error": MODEL_STATE["error"] or "Model not ready"}), 503

    body = request.get_json(force=True) or {}
    feature_cols = MODEL_STATE["feature_cols"]
    encoders = MODEL_STATE["encoders"]

    row = dict(FEATURE_DEFAULTS)
    for key, default in FEATURE_DEFAULTS.items():
        if key in body and body[key] is not None:
            row[key] = type(default)(body[key])

    shipping = body.get("shipping_method", "Standard")
    if "shipping_method" in encoders:
        encoder = encoders["shipping_method"]
        if shipping not in set(encoder.classes_):
            shipping = encoder.classes_[0]
        row["shipping_method"] = int(encoder.transform([shipping])[0])

    for col, encoder in encoders.items():
        if col == "shipping_method":
            continue
        value = str(body.get(col, encoder.classes_[0]))
        if value not in set(encoder.classes_):
            value = encoder.classes_[0]
        row[col] = int(encoder.transform([value])[0])

    db = get_db()
    prediction = {
        **row,
        "created_at": datetime.utcnow(),
        "source": "dashboard",
    }
    prediction_id = db[PREDICTION_COLLECTION].insert_one(prediction).inserted_id
    X = pd.DataFrame([row])[feature_cols]
    model = MODEL_STATE["model"]
    delayed = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    delay_prob = float(proba[classes.index(1)]) if 1 in classes else float(delayed)
    if delay_prob >= 0.75:
        risk_level = "High Risk of Delay"
    elif delay_prob >= 0.5:
        risk_level = "Moderate Risk of Delay"
    else:
        risk_level = "Low Risk of Delay"
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )[:6]
    return jsonify({
        "prediction_id": str(prediction_id),
        "delayed": bool(delayed),
        "risk_level": risk_level,
        "delay_probability": round(delay_prob * 100, 1),
        "on_time_probability": round((1 - delay_prob) * 100, 1),
        "importances": [{"name": n, "val": round(float(v) * 100, 1)} for n, v in importances],
    })


@app.route("/api/prediction-feedback", methods=["POST"])
def prediction_feedback():
    body = request.get_json(force=True) or {}
    prediction_id = body.get("prediction_id")
    actual_is_delayed = body.get("actual_is_delayed")
    if not prediction_id or actual_is_delayed not in (0, 1, False, True):
        return jsonify({
            "error": "prediction_id and actual_is_delayed (0 or 1) are required",
        }), 400

    try:
        object_id = ObjectId(str(prediction_id))
    except (InvalidId, TypeError):
        return jsonify({"error": "Invalid prediction_id"}), 400

    db = get_db()
    result = db[PREDICTION_COLLECTION].update_one(
        {
            "_id": object_id,
            "actual_is_delayed": {"$exists": False},
        },
        {
            "$set": {
                "actual_is_delayed": int(bool(actual_is_delayed)),
                "feedback_at": datetime.utcnow(),
            },
        },
    )
    if result.matched_count == 0:
        return jsonify({"error": "Prediction not found or feedback already recorded"}), 404

    labeled_count = db[PREDICTION_COLLECTION].count_documents(
        {"actual_is_delayed": {"$exists": True}},
    )
    retrained = False
    if labeled_count % 100 == 0:
        train_model()
        retrained = MODEL_STATE["ready"]

    return jsonify({
        "ok": True,
        "labeled_outcomes": labeled_count,
        "retrained": retrained,
        "model": MODEL_STATE["metrics"],
    })


train_model()

if __name__ == "__main__":
    print("Training on ecommerce_logistics_delivery.csv with feature importance...")
    if MODEL_STATE["ready"]:
        print("Model ready:", MODEL_STATE["metrics"])
    else:
        print("Model not ready:", MODEL_STATE["error"])
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
