import requests


BASE_URL = 'https://www.finn.no/'
SEARCH_PATH = 'recommerce/forsale/search'

search_params = {
    'q': 'gitar',
    'page' : 1,
}

res = requests.get(url = f"{BASE_URL}{SEARCH_PATH}", params=search_params, timeout = 15 )

print("Status code: ",res.status_code)

res.raise_for_status() # Stops the code if an error occurs when contacting the page

with open("data/finn_html_response.txt", mode='w') as f:
    f.write(res.text)