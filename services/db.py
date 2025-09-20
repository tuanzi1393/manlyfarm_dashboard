# services/db.py
from pymongo import MongoClient
import os

_client = None

def get_db():
    global _client
    if _client is None:
        # 默认连接到打包的本地 MongoDB
        mongo_port = os.environ.get("MONGO_PORT", "27017")
        _client = MongoClient(f"mongodb://localhost:{mongo_port}/")
    return _client["manly_farm"]
