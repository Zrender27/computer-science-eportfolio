"""
CS 499 Database Enhancement - Animal Shelter CRUD Module
Zoe Render

This enhanced version builds on the reconstructed CS 340 Animal Shelter
CRUD module. The enhancement focuses on secure configuration, input
validation, query limits, and complete CRUD functionality.
"""

import os
from typing import Dict, List, Optional

from pymongo import MongoClient


class AnimalShelter:
    """CRUD operations for the Animal collection in MongoDB."""

    # ENHANCEMENT 1:
    # Required fields are centralized so all new animal records are
    # validated consistently before they are written to the database.
    REQUIRED_FIELDS = {"animal_id", "animal_type", "breed", "name"}

    def __init__(self, client=None):
        """Create the database connection.

        A MongoClient can be injected for testing. When no client is passed,
        connection settings are read from environment variables.
        """

        # ENHANCEMENT 2:
        # Removed the password from source code. Sensitive credentials are
        # now read from environment variables instead of being hard-coded.
        user = os.getenv("AAC_USER", "aacuser")
        password = os.getenv("AAC_PASSWORD")
        host = os.getenv("AAC_HOST", "localhost")
        port = int(os.getenv("AAC_PORT", "27017"))
        database_name = os.getenv("AAC_DB", "aac")
        collection_name = os.getenv("AAC_COLLECTION", "animals")

        if client is None:
            if not password:
                raise EnvironmentError(
                    "AAC_PASSWORD environment variable must be set before connecting."
                )

            self.client = MongoClient(
                f"mongodb://{user}:{password}@{host}:{port}",
                serverSelectionTimeoutMS=5000,
            )
        else:
            # ENHANCEMENT 3:
            # Client injection makes the database layer easier to test and
            # removes the need to connect to production data during testing.
            self.client = client

        self.database = self.client[database_name]
        self.collection = self.database[collection_name]

    @classmethod
    def _validate_new_animal(cls, data: Dict) -> None:
        """Validate a new animal document before insertion."""

        # ENHANCEMENT 4:
        # The original create method only checked for None. This validation
        # rejects empty records and records that are missing required fields.
        if not isinstance(data, dict) or not data:
            raise ValueError("Animal data must be a non-empty dictionary.")

        missing = cls.REQUIRED_FIELDS - data.keys()
        if missing:
            raise ValueError(
                "Missing required field(s): " + ", ".join(sorted(missing))
            )

    def create(self, data: Dict) -> bool:
        """Insert one validated animal document."""
        self._validate_new_animal(data)
        result = self.collection.insert_one(data)
        return bool(result.acknowledged)

    def read(self, query: Optional[Dict] = None, limit: int = 50) -> List[Dict]:
        """Return matching records with a safe result limit."""

        # ENHANCEMENT 5:
        # Added query validation and a result limit so an empty query cannot
        # accidentally load an unbounded number of database records.
        if query is not None and not isinstance(query, dict):
            raise ValueError("Query must be a dictionary or None.")

        if not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("Limit must be an integer between 1 and 200.")

        cursor = self.collection.find(query or {}).limit(limit)
        return list(cursor)

    def update(self, query: Dict, updates: Dict) -> int:
        """Update records matching a filter and return the modified count."""

        # ENHANCEMENT 6:
        # Added the Update portion of CRUD. A non-empty filter is required to
        # reduce the risk of unintentionally changing every record.
        if not isinstance(query, dict) or not query:
            raise ValueError("Update requires a non-empty query filter.")
        if not isinstance(updates, dict) or not updates:
            raise ValueError("Updates must be a non-empty dictionary.")
        if "_id" in updates:
            raise ValueError("The MongoDB _id field cannot be modified.")

        result = self.collection.update_many(query, {"$set": updates})
        return int(result.modified_count)

    def delete(self, query: Dict) -> int:
        """Delete records matching a filter and return the deleted count."""

        # ENHANCEMENT 7:
        # Added the Delete portion of CRUD. Requiring a non-empty query helps
        # prevent an accidental deletion of the entire collection.
        if not isinstance(query, dict) or not query:
            raise ValueError("Delete requires a non-empty query filter.")

        result = self.collection.delete_many(query)
        return int(result.deleted_count)

    def close(self) -> None:
        """Close the MongoDB client connection."""

        # ENHANCEMENT 8:
        # Added explicit cleanup so the application can release the database
        # connection when work is complete.
        self.client.close()
