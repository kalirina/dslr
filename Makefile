PYTHON = .venv/bin/python
VENV = .venv

.PHONY: all install clean

all: install

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

fclean:
	rm -rf $(VENV)
