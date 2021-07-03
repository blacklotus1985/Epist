# importing the requests library
import requests
def correct_letter(text,URL="http://epistolarita-develop.kube.simultech.it/spellcheck"):
    """
    corrects letter using spellchecker
    :param text: text to correct
    :param URL: url of post call
    :return: corrected text
    """
    dict  = {"transcription":text}
    response = requests.post(url=URL, json=dict)
    return response.json()['translation']


