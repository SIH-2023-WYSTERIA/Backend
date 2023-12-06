import requests
from flask import jsonify

def model_proxy(audio_file):
    api_url = 'https://api.example.com/analyze_audio'

    files = {'audio': audio_file}

    response = requests.post(api_url, files=files)

    api_response = response.json()

    return jsonify(api_response)