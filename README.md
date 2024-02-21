### LANGCHAIN PGVECTOR OLLAMA EMBEDDINGS

# Here are files that are the squeleton of pgvector so postgresql enabled embeddings
# Ollama is used as local model loader
# lanchain is used for the python codebase as it has different interesting handles already made
# possibility to visiualise runs through langsmith

# requirement.txt is saved, script file and examples of text embeddings

#  full_speech_conversation.txt

Make a virtual environment
```python
python3 -m venv <NAME_OF_YOUR_VIRTUAL_ENVIRONMNET>
```
then activate it
```bash
source <NAME_OF_YOUR_VIRTUAL_ENVIRONMENT>/bin/activate
```

Your will need al so:
-pyaudio/
- intall postgresql compatible vector extension to enable embedding storage: via pg_embedding or lantern or  pgvector(the one i am using)
