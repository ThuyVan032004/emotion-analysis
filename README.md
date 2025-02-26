# Emotion Analysis API

Emotion Analysis on emotions: sad, joy, love, anger, fear, surprise

# Prerequisites
* [Python 3.10](https://www.python.org/downloads/release/python-31011/)

## Installation
1. Clone the repository
```bash
git clone https://github.com/ThuyVan032004/emotion-analysis.git
```
2. Add project root directory to PYTHONPATH
```bash
export PYTHONPATH=$PYTHONPATH:$PWD
```
3. Create virtual environment (recommend)
```bash
python -m venv .venv
```
4. Install the required packages
```bash
pip install -r requirements.txt
```
5. Run workloads
```bash
python -m emotion_analysis_svm.svm_workloads
```
6. Run API
* For development mode
```bash
bentoml serve
```

* For deployment mode

You will need to create a [bentoml account](https://www.bentoml.com/). Then create API token use the command
```bash
bentoml cloud login
```
Choose "Create a new API token with a web browser"

After that, you can deploy the API using command
```bash
bentoml deploy .
```
## Usage
* For development mode
```python 
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000/predict") as client:
    result = client.predict(
        input_text="Everyone likes the film",
    )
```
You can also use Postman Desktop to call the API

* For deployment mode
```python
import bentoml

with bentoml.SyncHTTPClient("https://<Your deployed API endpoint>") as client:
    result = client.predict(
        input_text="Everyone likes the film",
    )
```