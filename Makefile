# Configurações do ambiente
VENV = .venv
PYTHON = $(VENV)\Scripts\python.exe
PIP = $(VENV)\Scripts\pip.exe

# Lista de pastas e arquivos para análise
SOURCES = main.py config.py src\

.PHONY: all venv install format lint run clean

all: install format lint

# Criação do ambiente virtual
venv:
	@if not exist $(VENV) ( \
		python -m venv $(VENV) && \
		echo Ambiente virtual criado. \
	) else ( \
		echo Ambiente virtual já existe. \
	)

# Instalação de dependências e ferramentas de qualidade
install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install black isort flake8 mypy
	@echo Dependências instaladas com sucesso.

# Formatação automática (Correção)
format:
	@echo Formatando código com black e isort...
	$(PYTHON) -m black $(SOURCES)
	$(PYTHON) -m isort $(SOURCES)

# Sanitização (Linting e Checagem de Tipos)
lint:
	@echo Executando linting com flake8...
	$(PYTHON) -m flake8 $(SOURCES) --max-line-length=120 --ignore=E203,W503
	@echo Executando checagem de tipos com mypy...
	$(PYTHON) -m mypy $(SOURCES) --ignore-missing-imports

# Rodar o projeto (CLI)
run:
	$(PYTHON) main.py

# Rodar o projeto (Web)
web:
	$(VENV)\Scripts\streamlit run app.py --server.address 0.0.0.0

# Limpeza do ambiente
clean:
	@if exist $(VENV) rd /s /q $(VENV)
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@echo Ambiente limpo.
