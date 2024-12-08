#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = zp_ihlt_project
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# PATHS                                                                         #
#################################################################################

PROJECT_DIR = project
SRC_DIR = src
DATA_DIR = data

STAMPS_DIR = .stamps

SUBMISSION_NAME = ZacharyParent_2024_Project_IHLT
ZIP_NAME = $(SUBMISSION_NAME).zip

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf $(STAMPS_DIR)

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 $(SRC_DIR)/*
	isort --check --diff --profile black $(SRC_DIR)/*
	black --check --config pyproject.toml $(SRC_DIR)/*

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml $(SRC_DIR)/*

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then \
		source `which virtualenvwrapper.sh`; \
		mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); \
	else \
		mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); \
	fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"

## Create distribution zip
.PHONY: zip
zip:
	rm -f $(ZIP_NAME) 
	cd .. && \
	mkdir -p $(SUBMISSION_NAME) && \
	cp -r $(PROJECT_DIR)/* $(SUBMISSION_NAME)/ && \
	zip -r $(SUBMISSION_NAME).zip $(SUBMISSION_NAME) \
			-x "*/__pycache__/*" "*.pyc" \
			"*.aux" "*.log" "*.out" "*.toc" "*.bbl" "*.blg" \
			"*.fls" "*.nav" "*.snm" "*.vrb" "*.fdb_latexmk" "*.synctex.gz" \
			"*/.venv/*" "*/.git/*" && \
	rm -rf $(SUBMISSION_NAME) && \
	mv $(SUBMISSION_NAME).zip $(PROJECT_DIR)/  

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
