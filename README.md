# Extract-inator 🏭

## Instalação

Para começar, siga os passos abaixo ou execute a sequência de comandos a seguir:

```bash
pip install pipx && pipx ensurepath && pipx install poetry && poetry config virtualenvs.in-project true && poetry shell
```

### Passos de Instalação Detalhados

#### Instalação do pipx

```bash
pip install pipx 
```

#### Configuração do pipx

```bash
pipx ensurepath 
```

#### Instalação do Poetry

```bash
pipx install poetry
```

#### Configuração do Poetry

```bash
poetry config virtualenvs.in-project true
```

#### Criando e Ativando o Ambiente Virtual

```bash
poetry shell
```

### Configuração do VS Code

Se o VS Code não reconhecer automaticamente o ambiente virtual criado, siga estas etapas:

1. Abra as configurações do VS Code (`Ctrl + ,`).
2. Procure por `python.envFile` no campo de busca e altere para o nome do ambiente virtual corretamente criado.

### Instalando Dependências

```bash
poetry install
```

## Adicionando Novas Bibliotecas

Para adicionar uma nova biblioteca ao seu projeto, utilize o comando abaixo:

```bash
poetry add nome_da_biblioteca
```

---