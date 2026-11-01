# Smart Delivery Fleet Delay & Risk Predictor

## Introduction
This repository implements an end‑to‑end data pipeline that showcases **MongoDB** capabilities (schema validation, embedding, referencing, and aggregation) together with a **Scikit‑Learn** machine‑learning model for predicting delivery delays.

## Project Objective
- Store the data in MongoDB using best‑practice schema design.
- Perform advanced analytical queries with aggregation pipelines.
- Train and evaluate a classification model that predicts whether an order will be delayed.

## Dataset Description
The source CSV (`ecommerce_logistics_delivery.csv`) contains the following columns:
| Column | Description |
|--------|-------------|
| `order_id` | Unique order identifier (e.g., `ORD-123456`). |
| `driver_id` | Identifier that references a document in the `drivers` collection. |
| `distance_km` | Route distance in kilometres. |
| `traffic_index` | Multiplier representing traffic congestion (higher = more traffic). |
| `weather_condition` | Categorical weather description (`Sunny`, `Rainy`, `Snow`, `Fog`, `Windy`). |
| `package_weight_kg` | Weight of the package. |
| `delivery_status` | Enum: `Delivered`, `In‑Transit`, `Delayed`. |
| `is_delayed` | Target label (`0` = on‑time, `1` = delayed). |
| `checkpoint_logs` | Embedded array of checkpoint sub‑documents (timestamp, location, status). |

## MongoDB Schema Design & Validation Rules
- **Collections**: `drivers` (simple reference collection) and `orders` (main dataset).
- **JSON Schema Validation** on `orders` ensures:
  - `package_weight_kg` is a non‑negative double.
  - `delivery_status` matches the allowed enum values.
  - `is_delayed` is an integer in `{0,1}`.
  - `checkpoint_logs` is an array of sub‑documents each containing a `timestamp`, `location`, and `status`.
- **Referencing**: `orders.driver_id` → `drivers.driver_id`.
- **Embedding**: `checkpoint_logs` are stored directly inside each order document.

## PyMongo CRUD Operations (`database.py`)
1. **Connection** – `MongoClient` to a local MongoDB instance.
2. **Collection setup** – creates `drivers` and `orders` with the JSON schema validator.
3. **CSV ingestion** –
   - Upserts driver documents (one per unique `driver_id`).
   - Inserts order documents with embedded checkpoint logs.
4. **Indexes** – unique index on `order_id` and an index on `driver_id` for fast look‑ups.

## Aggregation Pipelines (`queries.py`)
1. **Metrics per delivery status** – `$group` to compute average distance, total weight, and order count.
2. **Driver enrichment** – `$lookup` + `$unwind` to join `orders` with the `drivers` collection.
3. **Faceted analysis** – `$facet` to simultaneously compute:
   - Top drivers by delay rate.
   - Weather conditions that cause the most delays.

## Machine Learning Implementation (`ml_model.py`)
  - Uses only information available before delivery starts; outcome-related fields and cost/estimate fields are excluded to reduce leakage.
- Splits data (80/20) with stratification.
- Supports two classifiers:
    - **Random Forest** (default, regularized with limited depth and minimum leaf sizes).
  - **Logistic Regression** (optional via `model_type` argument).
- Prints standard classification metrics: Accuracy, Precision, Recall, and F1‑Score.
- The dashboard stores every prediction input in MongoDB. After the actual delivery outcome is submitted, each labeled record becomes available for retraining; the model retrains after every 100 labeled outcomes.

## Installation
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install required packages
pip install pymongo pandas scikit-learn
```
> **Note:** The scripts use only standard library modules besides the three dependencies above.

## Usage
```bash
# 1. Ingest data into MongoDB
python database.py

# 2. Run aggregation examples
python queries.py

# 3. Train & evaluate the ML model
python ml_model.py   # use "--model_type logistic_regression" to switch models
```
All scripts print clear console messages indicating success.

## Deploy on Render with MongoDB Atlas

1. Push this project to GitHub and create a Render **Web Service** from the repository.
2. Render will use `requirements.txt` and start the app with Gunicorn.
3. In MongoDB Atlas, create a database user and allow the Render service to connect. For a quick public deployment, add `0.0.0.0/0` to Atlas Network Access; restrict this to Render's outbound IPs when available.
4. Add these Render environment variables:
   - `MONGO_URI`: your MongoDB Atlas connection string
   - `MONGO_DB_NAME`: `delivery_fleet`
5. Before opening the dashboard, run the ingestion once against Atlas from a trusted machine:

```powershell
$env:MONGO_URI = "mongodb+srv://<user>:<password>@<cluster>/<database>"
python database.py
```

6. Open the Render URL. The service trains from `ecommerce_logistics_delivery.csv` at startup. Prediction inputs and labeled outcomes are stored in the `prediction_inputs` collection, and the model retrains after every 100 labeled outcomes.

Never commit the Atlas connection string or database credentials to the repository.

## Project Structure
```
mongo bd project/
├── database.py           # MongoDB connection & ingestion
├── queries.py            # aggregation pipeline examples
├── ml_model.py           # ML training & evaluation
├── ecommerce_logistics_delivery.csv # source dataset
└── README.md             # **this** document
```

## Extending the Project
- Add more features (e.g., one‑hot encode `weather_condition`).
- Replace the source CSV with real order data.
- Deploy a lightweight API (FastAPI/Flask) that serves real‑time delay predictions.
- Automate nightly data refresh using a cron job or the `/schedule` slash command.

---
*This repository is intended for educational purposes and demonstrates how MongoDB and machine learning can be combined for logistics analytics.*
