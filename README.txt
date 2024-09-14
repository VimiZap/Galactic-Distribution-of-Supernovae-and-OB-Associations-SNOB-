To install the required packages, first set up a virtual environment on your machine and install the project as a package.

to create venv: 
- run: python3 -m venv venv (linux)
- run: python -m venv venv (windows)

- venv is activated if it says (.venv) in the terminal
- make sure the Python interpreter is selected for the .venv and not the global one

to activate venv: 
source venv/bin/activate  # Unix/Linux/macOS
venv\Scripts\activate     # Windows

to deactivate venv:
deactivate

to remove venv:
- deactive it first
- run: rm -rf .venv (linux)
- run: rmdir /s /q venv (windows)



To install project*:
- have a setup.py located in project root
- from project root, run: 'pip install -e .'


Note if you are using VSCode, the .vscode/settings.json file automaically activates the venv. 