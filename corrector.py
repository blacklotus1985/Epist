# importing the requests library
import requests

# api-endpoint
URL = "http://epistolarita-develop.kube.simultech.it/spellcheck"

# location given here
text = "haver mancato la casa honorando signor"

# defining a params dict for the parameters to be sent to the API
PARAMS = {'transcription':text}

# sending get request and saving the response as response object
r = requests.get(url = URL, params = PARAMS)

# extracting data in json format
data = r.json()


# extracting latitude, longitude and formatted address
# of the first matching location

# printing the output
print(1)
