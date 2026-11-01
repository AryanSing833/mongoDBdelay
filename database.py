'''database.py
MongoDB connection, collection setup with $jsonSchema validation,
and ingestion of the real e-commerce logistics dataset.
'''

import csv
import os
import random
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


# ---------------------------------------------------------------------------
# MongoDB Connection
# ---------------------------------------------------------------------------

def get_client(uri: str | None = None) -> MongoClient:
    """Return a MongoClient. In production, use TLS and env-based credentials."""
    return MongoClient(
        uri or os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017"),
        serverSelectionTimeoutMS=5000,
    )


def get_database(client: MongoClient, db_name: str = "delivery_fleet") -> Database:
    return client[db_name]


# ---------------------------------------------------------------------------
# Collection Setup with JSON Schema Validation
# ---------------------------------------------------------------------------

def setup_collections(db: Database) -> Dict[str, Collection]:
    """Create `drivers` (reference) and `orders` (main) collections.

    Demonstrates:
    - $jsonSchema validation on the orders collection
    - Referencing: orders store a driver_id that points to drivers collection
    - Embedding: checkpoint_logs array is embedded inside each order
    """

    # ── Drivers (reference collection) ─────────────────────────────────────
    if "drivers" not in db.list_collection_names():
        db.create_collection("drivers")
    drivers = db["drivers"]
    drivers.create_index("driver_id", unique=True)

    # ── Orders with $jsonSchema validator ──────────────────────────────────
    order_schema = {
        "bsonType": "object",
        "required": [
            "order_id", "driver_id", "distance_km", "traffic_delay_minutes",
            "weather_delay_minutes", "package_weight_kg", "shipping_method",
            "delivery_status", "is_delayed", "checkpoint_logs",
        ],
        "properties": {
            "order_id":               {"bsonType": "string"},
            "driver_id":              {"bsonType": "string"},
            "distance_km":            {"bsonType": "double", "minimum": 0},
            "traffic_delay_minutes":  {"bsonType": "double", "minimum": 0},
            "weather_delay_minutes":  {"bsonType": "double", "minimum": 0},
            "package_weight_kg":      {"bsonType": "double", "minimum": 0},
            "shipping_method":        {"enum": ["Standard", "Express", "Economy", "Same Day"]},
            "delivery_status":        {"enum": ["On Time", "Delayed", "Severely Delayed", "Cancelled"]},
            "is_delayed":             {"bsonType": "int", "minimum": 0, "maximum": 1},
            "customer_region":        {"bsonType": "string"},
            "warehouse_region":       {"bsonType": "string"},
            "courier_rating":         {"bsonType": "double", "minimum": 0, "maximum": 5},
            "checkpoint_logs": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "required": ["timestamp", "location", "status"],
                    "properties": {
                        "timestamp": {"bsonType": "date"},
                        "location":  {"bsonType": "string"},
                        "status":    {"bsonType": "string"},
                    },
                },
            },
        },
    }

    if "orders" not in db.list_collection_names():
        db.create_collection("orders", validator={"$jsonSchema": order_schema})
    else:
        db.command("collMod", "orders", validator={"$jsonSchema": order_schema})

    orders = db["orders"]
    orders.create_index("order_id", unique=True)
    orders.create_index("driver_id")
    orders.create_index("customer_region")
    orders.create_index("delivery_status")

    return {"drivers": drivers, "orders": orders}


# ---------------------------------------------------------------------------
# Embedded checkpoint log generator
# ---------------------------------------------------------------------------

def generate_checkpoint_logs(num: int = 3) -> List[Dict]:
    """Generate embedded checkpoint logs to demonstrate document embedding."""
    base_time = datetime.now()
    locations = ["Warehouse Gate", "City Hub", "Local Depot", "Out for Delivery", "Delivered"]
    statuses  = ["Dispatched", "In-Transit", "At Hub", "Out for Delivery", "Arrived"]
    logs = []
    for i in range(num):
        logs.append({
            "timestamp": base_time + timedelta(hours=i * 4),
            "location":  locations[min(i, len(locations) - 1)],
            "status":    statuses[min(i, len(statuses) - 1)],
        })
    return logs


# ---------------------------------------------------------------------------
# CSV Ingestion — Real Dataset
# ---------------------------------------------------------------------------

def ingest_csv(
    csv_path: str,
    db: Database,
    driver_collection: Collection,
    order_collection: Collection,
    limit: int = 5000,
) -> None:
    """Ingest the real e-commerce logistics CSV into MongoDB.

    - Upserts driver documents (referencing pattern).
    - Inserts order documents with embedded checkpoint_logs.
    - Limits to `limit` rows to keep ingestion fast (set to None for all).
    """
    print(f"Starting ingestion from: {csv_path}")
    inserted = 0
    skipped  = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break

            # Derive a driver_id from Delivery_Partner field
            driver_id = row["Delivery_Partner"].replace(" ", "_").upper()

            # Upsert driver reference document
            driver_collection.update_one(
                {"driver_id": driver_id},
                {"$setOnInsert": {
                    "driver_id": driver_id,
                    "name":      row["Delivery_Partner"],
                    "region":    row["Warehouse_Region"],
                }},
                upsert=True,
            )

            # Map is_delayed: 0 if "On Time", 1 otherwise
            is_delayed = 0 if row["On_Time_Delivery_Flag"] == "1" else 1

            order_doc = {
                "order_id":               row["Order_ID"],
                "driver_id":              driver_id,
                "order_date":             row["Order_Date"],
                "product_category":       row["Product_Category"],
                "customer_region":        row["Customer_Region"],
                "warehouse_region":       row["Warehouse_Region"],
                "shipping_method":        row["Shipping_Method"],
                "payment_method":         row["Payment_Method"],
                "order_value":            float(row["Order_Value"]),
                "order_quantity":         int(row["Order_Quantity"]),
                "package_weight_kg":      float(row["Package_Weight_KG"]),
                "package_volume_cm3":     float(row["Package_Volume_CM3"]),
                "distance_km":            float(row["Distance_KM"]),
                "warehouse_processing_h": float(row["Warehouse_Processing_Hours"]),
                "traffic_delay_minutes":  float(row["Traffic_Delay_Minutes"]),
                "weather_delay_minutes":  float(row["Weather_Delay_Minutes"]),
                "courier_rating":         float(row["Courier_Rating"]),
                "courier_experience_yrs": float(row["Courier_Experience_Years"]),
                "warehouse_capacity_pct": float(row["Warehouse_Capacity_Utilization"]),
                "inventory_availability": float(row["Inventory_Availability"]),
                "weekend_flag":           int(row["Weekend_Flag"]),
                "holiday_flag":           int(row["Holiday_Flag"]),
                "promotion_flag":         int(row["Promotion_Flag"]),
                "estimated_delivery_h":   float(row["Estimated_Delivery_Hours"]),
                "promised_delivery_h":    int(row["Promised_Delivery_Hours"]),
                "delivery_delay_hours":   float(row["Delivery_Delay_Hours"]),
                "delivery_status":        row["Delivery_Status"],
                "is_delayed":             is_delayed,
                "cancellation_flag":      int(row["Cancellation_Flag"]),
                "return_flag":            int(row["Return_Flag"]),
                "shipping_cost":          float(row["Shipping_Cost"]),
                "logistics_cost":         float(row["Logistics_Cost"]),
                "customer_satisfaction":  float(row["Customer_Satisfaction_Score"]),
                "order_hour":             int(row["Order_Hour"]),
                "day_of_week":            int(row["Day_of_Week"]),
                "month":                  int(row["Month"]),
                # Embedded array — demonstrates document embedding
                "checkpoint_logs":        generate_checkpoint_logs(),
            }

            try:
                order_collection.insert_one(order_doc)
                inserted += 1
            except Exception as e:
                skipped += 1

    print(f"Ingestion complete: {inserted} inserted, {skipped} skipped (duplicates/validation errors).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = get_client()
    db = get_database(client)

    # Drop existing collections for a clean run
    print("Dropping existing collections...")
    db.drop_collection("orders")
    db.drop_collection("drivers")

    collections = setup_collections(db)
    csv_file = Path(__file__).parent / "ecommerce_logistics_delivery.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_file}")

    # Ingest first 5000 rows (change to None to ingest all 15000)
    ingest_csv(
        str(csv_file),
        db,
        collections["drivers"],
        collections["orders"],
        limit=5000,
    )
    print("Done. Database: delivery_fleet | Collections: orders, drivers")
