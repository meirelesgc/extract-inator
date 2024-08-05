import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embedding_function
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.callbacks import get_openai_callback
from datetime import datetime
CHROMA_PATH = "exam-helpler-v2/chroma"

PROMPT_TEMPLATE = """
Você é um assistente de extração de dados de laudos clinicos.
Use os seguinte documento e somente ele, para me devolver o que for solicitado.
Se você não sabe ou não consegue encontrar a informação, me avise.

Preencha os campos que conseguir:

{format_instructions}

---

Responda as questões com base no contexto a seguir:

{context}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def create_parser():
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_core.output_parsers import JsonOutputParser

    class PatientEvolution(BaseModel):
        data_evolucao: datetime = Field(
            description="Data da evolução do paciente, no formato dd/mm/aaaa."
        )
        estado_geral: str = Field(
            description="Descrição do estado geral do paciente."
        )
        tratamento_inicial: str = Field(
            description="Tratamento inicial recebido pelo paciente."
        )
        historico_clinico: str = Field(
            description="""
                Histórico clínico do paciente, incluindo doenças e
                medicações em uso.
                """
        )
        exame_fisico: dict = Field(
            description="""
                Resultados do exame físico do paciente, incluindo aspectos
                neurológicos, cardiovasculares, respiratórios, digestivos,
                renais, endócrinos, hematológicos e infecciosos.
                """,
            example={
                "neurologico": "RASS -2, ECG normal, sem déficit motor. Pupilas isocóricas e fotorreagentes.",
                "cardiovascular": "Hemodinâmica estável, frequência cardíaca entre 82-95 bpm, pressão arterial média de 69-117 mmHg.",
                "respiratorio": "Confortável em ventilação não invasiva (CN) 2 L, com sinais de pneumonia e broncograma em base esquerda.",
                "digestivo": "Dieta enteral permitida, leve elevação nas transaminases (TGO 38, TGP 44).",
                "renal": "Diurese de 1300 ml em 12h, com creatinina elevada (4,07).",
                "endocrino": "Glicemia entre 123-134 mg/dL, sem escapes.",
                "hematologico": "Anemia (Hb 8,2), sem sangramentos visíveis.",
                "infeccioso": "Afebril, com foco pulmonar e leucocitose (11.600)."
            }
        )

    return JsonOutputParser(pydantic_object=PatientEvolution)


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)
    # if len(results) == 0 or results[0][1] < 0.9:
    #     print("Sem embasamento o suficiente para responder.")
    #     return

    setup = [("system", PROMPT_TEMPLATE), ("user", "{input}")]
    prompt_template = ChatPromptTemplate.from_messages(setup)
    parser = create_parser()

    model = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
    question_answer_chain = create_stuff_documents_chain(
        llm=model, prompt=prompt_template, output_parser=parser)
    reg_chain = create_retrieval_chain(
        db.as_retriever(), question_answer_chain)

    with get_openai_callback() as cb:
        response_json = reg_chain.invoke({
            'input': query_text,
            'format_instructions': parser.get_format_instructions()
        })
        print(cb)

    from pprint import pprint

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    pprint(response_json['answer'])
    print(f"Fontes: {sources}")
    return response_json


if __name__ == "__main__":
    main()
