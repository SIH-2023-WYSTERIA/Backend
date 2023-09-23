# Specify Bash as the shell
SHELL := /bin/bash

# Define the name of your virtual environment
VENV_NAME = venv

# Define the activation command based on the OS
ifeq ($(OS),Windows_NT)
    VENV_ACTIVATE = $(VENV_NAME)/Scripts/activate
else
    VENV_ACTIVATE = . $(VENV_NAME)/bin/activate
endif

.PHONY: all
all: run

# Run main.py after checking if the virtual environment is active
run:
	@if [ -z "$$VIRTUAL_ENV" ]; then $(VENV_ACTIVATE) && python3 main.py; else python3 main.py; fi

# Create a virtual environment if it doesn't exist
venv:
	@if [ ! -d $(VENV_NAME) ]; then python3 -m venv $(VENV_NAME); fi

# Clean up the virtual environment
clean:
	rm -rf $(VENV_NAME)

# Help target to display available targets in the Makefile
help:
	@echo "Available targets:"
	@echo "  venv   - Create a virtual environment"
	@echo "  run    - Activate the virtual environment (if not active) and run main.py"
	@echo "  clean  - Remove the virtual environment"
	@echo "  help   - Display this help message"
