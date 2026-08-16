"""Simple test script for the enhanced Animal Shelter CRUD module.
Zoe Render
"""

from CRUD_Python_Module_Enhanced import AnimalShelter


# ENHANCEMENT TEST:
# This script demonstrates all four CRUD operations instead of testing only
# Create and Read as the original CS 340 script did.
shelter = AnimalShelter()

try:
    test_animal = {
        "age_upon_outcome": "2 years",
        "animal_id": "UNITTEST123",
        "animal_type": "Dog",
        "breed": "Pit Bull Mix",
        "color": "Brown",
        "name": "TestPup",
        "outcome_type": "Adoption",
        "outcome_subtype": "Foster",
        "sex_upon_outcome": "Neutered Male",
    }

    # CREATE
    print("Create successful:", shelter.create(test_animal))

    # READ - limited and filtered
    records = shelter.read({"animal_id": "UNITTEST123"}, limit=10)
    print("Records found:", len(records))

    # UPDATE
    updated = shelter.update(
        {"animal_id": "UNITTEST123"},
        {"outcome_subtype": "Adoption Event"},
    )
    print("Records updated:", updated)

    # DELETE
    deleted = shelter.delete({"animal_id": "UNITTEST123"})
    print("Records deleted:", deleted)

finally:
    # ENHANCEMENT TEST:
    # Always close the connection, even if a database operation raises an error.
    shelter.close()
