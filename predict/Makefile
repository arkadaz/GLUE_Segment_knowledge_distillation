install:
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 &&\
	pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C Glue_inference.py

test:
	python -m pytest -vv --cov=GLUE test_predict.py 

all: install format test lint