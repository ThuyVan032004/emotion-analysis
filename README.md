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

You will need to create a [bentoml account](https://www.bentoml.com/)
## Usage

```python

```

## Contributing
