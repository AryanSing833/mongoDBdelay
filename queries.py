'''queries.py
Advanced MongoDB aggregation pipelines on the real e-commerce logistics dataset.
'''

import os

from pymongo import MongoClient
from pprint import pprint


def get_client(uri: str | None = None) -> MongoClient:
    return MongoClient(
        uri or os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017"),
        serverSelectionTimeoutMS=5000,
    )


def run_aggregations(db_name: str | None = None):
    client = get_client()
    db = client[db_name or os.environ.get("MONGO_DB_NAME", "delivery_fleet")]
    orders = db["orders"]

    # -----------------------------------------------------------------------
    # Pipeline A — Group by delivery_status: avg distance, total orders,
    #              avg delay hours, avg customer satisfaction
    # -----------------------------------------------------------------------
    pipeline_a = [
        {"$group": {
            "_id":                   "$delivery_status",
            "order_count":           {"$sum": 1},
            "avg_distance_km":       {"$avg": "$distance_km"},
            "avg_delay_hours":       {"$avg": "$delivery_delay_hours"},
            "avg_satisfaction":      {"$avg": "$customer_satisfaction"},
            "total_logistics_cost":  {"$sum": "$logistics_cost"},
        }},
        {"$sort": {"avg_delay_hours": -1}},
    ]
    print("\n" + "="*60)
    print("Pipeline A: Metrics grouped by Delivery Status")
    print("="*60)
    for doc in orders.aggregate(pipeline_a):
        print(f"\n  Status           : {doc['_id']}")
        print(f"  Order Count      : {doc['order_count']}")
        print(f"  Avg Distance     : {doc['avg_distance_km']:.2f} km")
        print(f"  Avg Delay        : {doc['avg_delay_hours']:.2f} hours")
        print(f"  Avg Satisfaction : {doc['avg_satisfaction']:.2f} / 5")
        print(f"  Total Logist.Cost: ${doc['total_logistics_cost']:.2f}")

    # -----------------------------------------------------------------------
    # Pipeline B — $lookup: join orders with drivers collection
    #              Show driver name, region alongside order details
    # -----------------------------------------------------------------------
    pipeline_b = [
        {"$lookup": {
            "from":         "drivers",
            "localField":   "driver_id",
            "foreignField": "driver_id",
            "as":           "driver_info",
        }},
        {"$unwind": "$driver_info"},
        {"$project": {
            "_id":              0,
            "order_id":         1,
            "delivery_status":  1,
            "shipping_method":  1,
            "distance_km":      1,
            "driver_name":      "$driver_info.name",
            "driver_region":    "$driver_info.region",
        }},
        {"$limit": 5},
    ]
    print("\n" + "="*60)
    print("Pipeline B: Orders joined with Driver info ($lookup)")
    print("="*60)
    for doc in orders.aggregate(pipeline_b):
        pprint(doc)

    # -----------------------------------------------------------------------
    # Pipeline C — $facet: run multiple sub-analyses in one shot
    #   1. driver_performance: delay rate per delivery partner
    #   2. region_analysis:    avg delay hours & satisfaction per customer region
    #   3. shipping_analysis:  avg cost and delay per shipping method
    # -----------------------------------------------------------------------
    pipeline_c = [
        {"$facet": {
            "driver_performance": [
                {"$group": {
                    "_id":            "$driver_id",
                    "total_orders":   {"$sum": 1},
                    "delayed_orders": {"$sum": "$is_delayed"},
                    "avg_rating":     {"$avg": "$courier_rating"},
                }},
                {"$project": {
                    "driver_id":   "$_id",
                    "total_orders": 1,
                    "delayed_orders": 1,
                    "delay_rate":  {"$round": [{"$divide": ["$delayed_orders", "$total_orders"]}, 3]},
                    "avg_rating":  {"$round": ["$avg_rating", 2]},
                    "_id": 0,
                }},
                {"$sort": {"delay_rate": -1}},
                {"$limit": 5},
            ],
            "region_analysis": [
                {"$group": {
                    "_id":              "$customer_region",
                    "avg_delay_hours":  {"$avg": "$delivery_delay_hours"},
                    "avg_satisfaction": {"$avg": "$customer_satisfaction"},
                    "total_orders":     {"$sum": 1},
                }},
                {"$sort": {"avg_delay_hours": -1}},
            ],
            "shipping_analysis": [
                {"$group": {
                    "_id":              "$shipping_method",
                    "avg_delay_hours":  {"$avg": "$delivery_delay_hours"},
                    "avg_ship_cost":    {"$avg": "$shipping_cost"},
                    "total_orders":     {"$sum": 1},
                    "on_time_count":    {"$sum": {"$cond": [{"$eq": ["$is_delayed", 0]}, 1, 0]}},
                }},
                {"$project": {
                    "method":          "$_id",
                    "avg_delay_hours": {"$round": ["$avg_delay_hours", 2]},
                    "avg_ship_cost":   {"$round": ["$avg_ship_cost", 2]},
                    "total_orders":    1,
                    "on_time_rate":    {"$round": [{"$divide": ["$on_time_count", "$total_orders"]}, 3]},
                    "_id": 0,
                }},
                {"$sort": {"avg_delay_hours": 1}},
            ],
        }},
    ]

    print("\n" + "="*60)
    print("Pipeline C: $facet — Driver Performance, Region & Shipping Analysis")
    print("="*60)
    result = list(orders.aggregate(pipeline_c))
    if result:
        facets = result[0]
        print("\n-- Top 5 Drivers by Delay Rate --")
        for d in facets["driver_performance"]:
            pprint(d)
        print("\n-- Delay & Satisfaction by Customer Region --")
        for r in facets["region_analysis"]:
            pprint(r)
        print("\n-- Performance by Shipping Method --")
        for s in facets["shipping_analysis"]:
            pprint(s)


if __name__ == "__main__":
    run_aggregations()
