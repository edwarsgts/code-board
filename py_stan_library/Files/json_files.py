import json
# Working with JSON files
movies = [
    {"id": 1, "title": "Terminator", "year": 1989},
    {"id": 2, "title": "Terminators", "year": 1999}
]

data = json.dumps(movies)
print(data)
