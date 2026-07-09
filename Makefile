PYTHON = .venv/bin/python
VENV = .venv

.PHONY: all install clean

all: install

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

clean:
	rm -rf model.json houses.csv

clean-graphs:
	rm -rf BGD_loss.png SGD_loss.png Mini_batch_GD_loss.png

fclean:
	rm -rf $(VENV)
