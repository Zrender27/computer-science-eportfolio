"""
Reconstructed CS 340 Module Four test script.

This mirrors the visible notebook test from the submitted screenshots.
"""

from CRUD_Python_Module import AnimalShelter


# Instantiate an instance of the CRUD class
shelter = AnimalShelter()

# Use the create function to create a new record in the AAC database
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

insert_result = shelter.create(test_animal)
print("Insert successful?", insert_result)

# Use the read function to return records from the AAC database
results = shelter.read({"animal_id": "UNITTEST123"})
print("Number of records found:", len(results))

for doc in results:
    print(doc)
