install_cuda:
	sudo apt ubuntu-drivers devices &&\
	sudo ubuntu-drivers autoinstall

install:
ifeq ($(OS),Windows_NT)
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 && pip install -r requirements.txt
else
	pip3 install torch torchvision torchaudio && pip3 install -r requirements.txt
endif

format:
	black *.py

lint:
	pylint --disable=R,C Glue_inference.py

test:
	python -m pytest -vv --cov=GLUE test_predict.py 

all: install_cuda install format test