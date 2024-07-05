from flask import Flask, request, jsonify

url = 'http://127.0.0.1:5000/predict'
data = {
    "msPlayed": [210000],
    "genre": ["pop"],
    "danceability": [0.8],
    "energy": [0.7],
    "key": [5],
    "loudness": [-5],
    "mode": [1],
    "speechiness": [0.05],
    "acousticness": [0.1],
    "instrumentalness": [0],
    "liveness": [0.1],
    "valence": [0.8],
    "tempo": [120],
    "duration_sec": [210],
    "time_signature": [4]
}

response = request.post(url, json=data)
print(response.json())
