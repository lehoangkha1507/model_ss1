import requests

url = "https://model-ss1.onrender.com/predict"
data = {"features": [10, 40, 50, 60, 30, 10, 35]}

response = requests.post(url, json=data)
print("🔍 API Response Status Code:", response.status_code)
print("🔍 API Response JSON:", response.json())
