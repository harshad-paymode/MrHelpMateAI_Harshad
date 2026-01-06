from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from core.config import PATHS, MODELS
from langchain_core.documents import Document
import json
from uuid import uuid4

def create_collection():
    embeddings = MistralAIEmbeddings(model=MODELS.MODEL_EMBED)
    insurance_collection = Chroma(
    collection_name="collection_insurance",
    embedding_function=embeddings,
    persist_directory=PATHS.CHROMA_PERSISTENT,  # Where to save data locally, remove if not necessary
    )
    return insurance_collection

def index_and_store(chunks):
    # Add the documents and metadata to the collection alongwith generic integer IDs. You can also feed the metadata information as IDs by combining the policy name and page no.
    docs = [
        Document(
            page_content=chunk["Text"],
            metadata=chunk["Metadata"]
        )
        for chunk in chunks
    ]

    uuids = [str(uuid4()) for _ in range(len(docs))]
    #Get the insurance collecton object
    insurance_collection = create_collection()
    insurance_collection.add_documents(documents=docs, ids=uuids)
