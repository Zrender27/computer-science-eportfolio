"""
Reconstructed from CS 340 Module Four screenshots.

This file recreates the visible AnimalShelter CRUD module structure.
The password shown in the original screenshot has intentionally been
replaced with a placeholder. Insert the appropriate course-environment
credential only if you need to run this against that MongoDB instance.
"""

from pymongo import MongoClient
from bson.objectid import ObjectId


class AnimalShelter(object):
    """CRUD operations for Animal collection in MongoDB."""

    def __init__(self):
        # Initializing the MongoClient. This helps to access the MongoDB
        # databases and collections. This is hard-wired to use the AAC
        # database, the animals collection, and the AAC user.

        # Connection Variables
        USER = "aacuser"
        PASS = "YOUR_PASSWORD_HERE"
        HOST = "localhost"
        PORT = 27017
        DB = "aac"
        COL = "animals"

        # Initialize Connection
        self.client = MongoClient(
            f"mongodb://{USER}:{PASS}@{HOST}:{PORT}"
        )
        self.database = self.client[DB]
        self.collection = self.database[COL]

    # Create method to implement the C in CRUD.
    def create(self, data):
        """
        Insert a document into the animals collection.

        :param data: Non-empty dictionary of key/value pairs.
        :return: True if successful insert, else False.
        """
        if data is not None:
            try:
                result = self.collection.insert_one(data)
                return result.acknowledged
            except Exception as e:
                print("An exception occurred during insert:", e)
                return False
        else:
            raise Exception("Nothing to save, because data parameter is empty")

    # Create method to implement the R in CRUD.
    def read(self, query=None):
        """
        Query for documents from the animals collection using find().

        :param query: Dictionary of key/value pairs to use as a filter.
                      If None, all documents are returned.
        :return: List of documents, or an empty list if no results / error.
        """
        try:
            if query is None:
                cursor = self.collection.find()
            else:
                cursor = self.collection.find(query)

            return list(cursor)
        except Exception as e:
            print("An exception occurred during query:", e)
            return []
