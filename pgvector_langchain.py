from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv
from flask import Flask, request
from database import init_db, SessionLocal
from models import PgRecord
from flask import Flask, request, jsonify

### SET UP ENV VARS, FLASK APP, DATABASE INIT, OPENAIKEY

# load env vars
load_dotenv()
# create flask app
app = Flask(__name__)
# Initialize the database
init_db()
# Openai key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

### LANGCHAIN EMBEDDING AND RETRIVAL PART

# VAR; get doc/text and split in chunks cardano_meme_coin.txt, best_meme_coins_2024.txt, history_of_coins.txt
list_documents_txt = ["cardano_meme_coin.txt", "best_meme_coins_2024.txt", "history_of_coins.txt", "article.txt"]

# Use ollama to create embeddings
embeddings = OllamaEmbeddings()

# define connection to pgvector database
CONNECTION_STRING = PGVector.connection_string_from_db_params(
     driver=os.getenv("DRIVER"),
     host=os.getenv("HOST"),
     port=int(os.getenv("PORT")),
     database=os.getenv("DATABASE"),
     user=os.getenv("USER"),
     password=os.getenv("PASSWORD"),
)
# define collection name
COLLECTION_NAME = "chat_embeddings"


# HELPER functions , create collection, retrieve from collection, chunk documents
def chunk_doc(path: str, files: list) -> list:
  list_docs = []
  for file in files:
    loader = TextLoader(f"{path}/{file}")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    list_docs.append(docs)
  return list_docs

def vector_db_create(doc, collection, connection):
  db_create = PGVector.from_documents(
    embedding=embeddings,
    documents=doc,
    collection_name=collection, # must be unique
    connection_string=connection,
  )
  return db_create

def vector_db_retrieve(collection, connection, embedding):
  db_retrieve = PGVector(
    collection_name=collection,
    connection_string=connection,
    embedding_function=embedding,
  )
  return db_retrieve

def add_document_and_retrieve(content, store):
  store.add_documents([Document(page_content=f"{content}")])
  docs_with_score = db.similarity_search_with_score("{content}")
  return doc_with_score

def vector_db_override(docs, embedding, collection, connection):
  for doc in docs:
    changed_db = PGVector.from_documents(
      documents=doc,
      embedding=embedding,
      collection_name=collection,
      connection_string=connection,
      pre_delete_collection=True,
    )
  return changed_db


## Creation of the collection
all_docs = chunk_doc("/home/creditizens/voice_llm", ["article.txt"]) # list_documents_txt
def create_embedding_collection(all_docs: list) -> str:
  collection_name = COLLECTION_NAME
  connection_string = CONNECTION_STRING
  for doc in all_docs:
    vector_db_create(doc, collection_name, connection_string)
  return f"Collection created with len list of document: {len(all_docs)}"

# print(create_embedding_collection(all_docs))

##  similarity query
question = "What is the story of Azabujuuban ?"

def similarity_search(question):
  db = vector_db_retrieve(COLLECTION_NAME, CONNECTION_STRING, embeddings)
  docs_and_similarity_score = db.similarity_search_with_score(question)
  for doc, score in docs_and_similarity_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

## MMR (Maximal Marginal Relevance) query
def MMR_search(question):
  db = vector_db_retrieve(COLLECTION_NAME, CONNECTION_STRING, embeddings)
  docs_and_MMR_score = db.max_marginal_relevance_search_with_score(question)
  for doc, score in docs_and_MMR_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
#print(MMR_search("Any coliving concept?")[0])

## OR use ollama query embedding
#text = "How many needs are they in Chikara houses?"
#query_result = embeddings.embed_query(text)
#print(query_result[:3])

"""
## OR just the route for Ollama native without langchain embeddings
curl http://localhost:11434/api/embeddings -d '{
  "model": "mistral:7b",
  "prompt": "Here is an article about llamas..."
}'
Outputs:
{
  "embedding": [
    0.5670403838157654, 0.009260174818336964, 0.23178744316101074, -0.2916173040866852, -0.8924556970596313,
    0.8785552978515625, -0.34576427936553955, 0.5742510557174683, -0.04222835972905159, -0.137906014919281
  ]
}
"""

# add document to store and retrieving
#store_db = vector_db_retrieve(COLLECTION_NAME, CONNECTION_STRING, embeddings)
#doc1 = add_document_and_retrieve("foo", store_db)[0] # add_document_and_retrieve("foo")[1]

# if you want to override vector store need .from_documents and option pre_delete_collection=True
#files = ["article.txt"]
#docs = chunk_doc("/home/creditizens/voice_llm", files)
#override_embeddings = vector_db_override(docs, embeddings, COLLECTION_NAME, CONNECTION_STRING)

"""
#### LFASK APP ROUTES "see how to pass question and answers data through a post request from another function"
@app.route('/embedding_record_conversation/<question>/<answer>', methods=['POST'])
def embedding_record_conversation(question, answer):
    # Create a new session instance
    db = SessionLocal()
    try:
        data = request.json
        new_preference = UserPreference(**data)
        db.add(new_preference)
        db.commit()
        return jsonify({"message": "User preference added successfully"}), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 400
    finally:
        db.close()

## FLASK APP  START
if __name__ == "__main__":
    app.run(debug=True)
"""
