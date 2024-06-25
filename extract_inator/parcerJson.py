from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

modelo = ChatOpenAI(model="gpt-4", temperature=0)


# Define a estrutura de dados desejada.
class Piada(BaseModel):
    pergunta: str = Field(description="pergunta para iniciar uma piada")
    resposta: str = Field(description="resposta para concluir a piada")


# E uma consulta destinada a solicitar a um modelo de linguagem para preencher a estrutura de dados.
consulta_piada = "Conte-me uma piada."

# Configurar um analisador + injetar instruções no template de prompt.
analisador = JsonOutputParser(pydantic_object=Piada)

prompt = PromptTemplate(
    template="Responda à consulta do usuário.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": analisador.get_format_instructions()},
)

cadeia = prompt | modelo | analisador

print(cadeia.invoke({"query": consulta_piada}))
