import os
os.environ['CURL_CA_BUNDLE'] = ''
PROXY = {"http": "http://localhost:3122", "https": "http://localhost:3122"}

# Monkey patch the requests library to disable SSL verification
def no_ssl_verification():
    import requests.adapters
    import urllib3

    # Ignore SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    original_send = requests.adapters.HTTPAdapter.send

    def send(*args, **kwargs):
        kwargs['verify'] = False
        return original_send(*args, **kwargs)

    requests.adapters.HTTPAdapter.send = send

no_ssl_verification()

from transformers import GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import os

def download_model(model_name, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Download the tokenizer and model
    if "gpt-j" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, proxies=PROXY)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, proxies=PROXY)
    model = AutoModelForCausalLM.from_pretrained(model_name, proxies=PROXY)
    
    # Save the tokenizer and model
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

# Define directories for the models
save_dir_gpt2 = './models/pt_models/gpt2-small'
save_dir_gpt2_xl = './models/pt_models/gpt2-xl'


# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

# Download and save the models
download_model('gpt2', save_dir_gpt2)
download_model('gpt2-xl', save_dir_gpt2_xl)