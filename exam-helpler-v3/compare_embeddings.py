from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv

load_dotenv()


def main():
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = (str(input('primeiro termo: ')), str(input('primeiro termo: ')))
    x = evaluator.evaluate_string_pairs(
        prediction=words[0], prediction_b=words[1])
    print(f"comparando ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query("virtual")
    print(f"vetor da palavra 'virtual': {vector}")
    print(f"tamanho do vetor: {len(vector)}")
    while True:
        main()
