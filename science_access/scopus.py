import requests
import json

# from my_scopus import MY_API_KEY
MY_API_KEY = "6fae5b65e32e1c403792a2c7301ebfe7"

resp = requests.get(
    "http://api.elsevier.com/content/author?author_id=7004212771&view=metrics",
    headers={"Accept": "application/json", "X-ELS-APIKey": MY_API_KEY},
)

print(json.dumps(resp.json(), sort_keys=True, indent=4, separators=(",", ": ")))
