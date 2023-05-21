install:
ifeq ($(OS),Windows_NT)
	sudo apt update && sudo apt-get install cuda-11-7 && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 && pip install -r requirements.txt
else
	sudo apt update && sudo apt-get install cuda-11-7 && pip3 install torch torchvision torchaudio && pip3 install -r requirements.txt
endif

format:
	black *.py

lint:
	pylint --disable=R,C Glue_inference.py

test:
	python -m pytest -vv --cov=GLUE test_predict.py 

all: install format test