import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

import json
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class GeminiEmbeddingWrapper(Embeddings):
    # embed_documents(texts: List[str]) -> List[List[float]] | Returns list of embeddings from text in docs
    def embed_documents(self, texts):
        return [
            genai.embed_content(model="models/embedding-001", content=t)["embedding"]
            for t in texts
        ]

    # embed_query(text: str) -> List[float] |
    def embed_query(self, text):
        return self.embed_documents([text])[0]


def prep_data(path):  # Takes in path to data, returns list[Document]
    data_path = path  # the name(or path if in different directory) of your data folder
    data_sub_path = os.path.join(data_path, os.listdir(data_path)[0])  # make cleaner

    # Loading and splitting Documents #

    # Building list of paths
    paths = [os.path.join(data_sub_path, path) for path in os.listdir(data_sub_path)]

    # Loads docs | loader.load() returns a LangChain document object, this holds content and metadata like sources
    single_path = paths[0]  # placeholder pre-for loop or other ease of access
    loader = PyPDFLoader(paths[0])  # either change to forloop or have user input decide
    pages = loader.load()

    # Join content while recording char offsets per page
    full_text = ""
    page_offsets = []
    for page in pages:
        start = len(full_text)
        full_text += page.page_content + "\n"
        end = len(full_text)
        page_offsets.append((start, end))  # (char_start, char_end) for each page

    # Treats the PDF as a single document without page splits
    docs = [Document(page_content=full_text, metadata={"source": single_path})]

    # Splits the loaded docs into chunks | splitter.split_documents() expects a LangChain document object
    # RecursiveCharacterTextSplitter is a robust option for preserving sentence structure
    # splitter.split_documents() returns a list[Document]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        full_text_start = full_text.find(chunk.page_content)
        full_text_end = full_text_start + len(chunk.page_content)
        page_range = find_pages_for_span(full_text_start, full_text_end, page_offsets)
        chunk.metadata["pages"] = page_range

    return chunks

    # print('Page content: ', str(chunks[50].page_content), "\n Pages: ", str(chunks[50].metadata["pages"]))


def build_database(chunks, data_dump='VecDB', save=True):  # builds the embedding database and saves it to data_dump
    embedding_mod = GeminiEmbeddingWrapper()

    if not save:
        data_dump = None

    # Originally document based, but page conservation caused problems so metadata must be cleaned
    texts = [chunk.page_content for chunk in chunks]
    # ensure each metadata value is a primitive
    metadatas = [clean_metadata(chunk.metadata) for chunk in chunks]

    vectordatabase = Chroma.from_texts(
        texts=texts,
        embedding=embedding_mod,
        metadatas=metadatas,
        persist_directory=data_dump
    )


def load_database(data_dir):
    vectordatabase = Chroma(
        persist_directory=data_dir,
        embedding_function=GeminiEmbeddingWrapper()
    )
    return vectordatabase


# Map each chunk back to page range | Helper function for finding which pages correspond to which chunks
def find_pages_for_span(start, end, page_offsets):
    pages_in_chunk = []
    for i, (p_start, p_end) in enumerate(page_offsets):
        if start < p_end and end > p_start:
            pages_in_chunk.append(i + 1)  # 1-indexed pages
    return pages_in_chunk


def clean_metadata(raw_metadata):  # The page tracking added non-parsable metadata, this cleans it while preserving info
    """
    Convert any list-valued metadata into a Chroma-compatible primitive.
    - If it’s a single-element list of a primitive, unwrap it.
    - Otherwise JSON-serialize the list.
    Other types get passed through if already primitive, or str-coerced as a fallback.
    """
    clean = {}
    for key, value in raw_metadata.items():
        # Allowed primitives
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        # Handle lists
        elif isinstance(value, list):
            if len(value) == 1 and (isinstance(value[0], (str, int, float, bool)) or value[0] is None):
                clean[key] = value[0]
            else:
                clean[key] = json.dumps(value)
        # Fallback for anything else
        else:
            clean[key] = str(value)
    return clean


def main():
    chunks = prep_data('data')
    build_database(chunks)


if __name__ == "__main__":
    main()
