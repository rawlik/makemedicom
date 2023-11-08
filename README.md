# makemedicom

## Installation

### Command-line interface
First install pipx: https://pypa.github.io/pipx/
Pipx will install makemedicom in its own dedicated virtual environment
so that you can use it without it interfering with

Install the latest release:
```bash
pipx install makemedicom
```

Install the development version straight from the repository:
```bash
pipx install git+https://github.com/rawlik/makemedicom.git

### Python module
Install the latest release in the desired environment:
```bash
pip install makemedicom
```

Install the development version straight from the repository:
```bash
pip install git+https://github.com/rawlik/makemedicom.git
```

## Usage
```bash
makemedicom myfile.h5
```

## Development

### CLI
```bash
git clone git@github.com:rawlik/makemedicom.git
cd makemedicom
pipx install --force --editable .
```

### Python module
If you need a dedicated environment for development:
```bash
python3 -m venv makemedicomdevenv
```
And then source the activation script appropriate for the shell.
For bash or zsh:
```bash
source makemedicomdevenv/env/bin/activate
```

In the desired environment run:
```bash
git clone git@github.com:rawlik/makemedicom.git
cd makemedicom
pip install --force --editable .
```
