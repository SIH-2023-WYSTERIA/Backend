# Define the name of your virtual environment
VENV_NAME = venv

# Define the activation command based on the OS
ifeq ($(OS),Windows_NT)
    VENV_ACTIVATE = $(VENV_NAME)/Scripts/activate
else
    VENV_ACTIVATE = . $(VENV_NAME)/bin/activate
endif

.PHONY: all
all: venv run

# Create a virtual environment
venv:
	python3 -m venv $(VENV_NAME)

# Activate the virtual environment and run main.py
run: venv
	$(VENV_ACTIVATE) && python3 main.py

# Clean up the virtual environment
clean:
	rm -rf $(VENV_NAME)

# Help target to display available targets in the Makefile
help:
	@echo "Available targets:"
	@echo "  venv   - Create a virtual environment"
	@echo "  run    - Activate the virtual environment and run main.py"
	@echo "  clean  - Remove the virtual environment"
	@echo "  help   - Display this help message"

