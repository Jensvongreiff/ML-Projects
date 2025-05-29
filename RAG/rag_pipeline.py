import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from google import genai
import database

import json
from dotenv import load_dotenv

load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

data_dir = 'VecDB'
default_input = 'Tell me about the knowledgebase i provided'

client = genai.Client()


def generate_answer(k=3, input_query=True, return_context=True):
    vectordatabase = database.load_database(data_dir=data_dir)

    if input_query:
        query = input('Input your prompt: ')
    else:
        query = default_input

    results = vectordatabase.similarity_search(query, k)

    if return_context:
        print("\n--- Retrieved Context ---")
        for i, doc in enumerate(results):
            print(f"[{i+1}] {doc.page_content[:200]}...\n")

    context = "\n\n".join(doc.page_content for doc in results)
    prompt = f"""Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    print("\n--- Gemini Response ---")
    print(str(response.text))

if __name__ == "__main__":
    # results = vectordatabase.similarity_search_with_score(query, k=3)
    # for doc, score in results:
    #     print("Score: ", str(score), "\nContent: ", str(doc.page_content))
    # database.main()
    generate_answer(k=5)


