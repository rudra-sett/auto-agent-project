import os
import subprocess
import sys
import importlib
import json
import requests
from transformers import AutoTokenizer

# Global variable to track the current working directory
current_directory = os.getcwd()

# Registry mapping tool names to functions
tool_registry = {}

def register_tool(func):
    """Decorator to register a tool in the registry."""
    tool_registry[func.__name__] = func
    return func

@register_tool
def change_directory(directory):
    global current_directory
    try:
        if not os.path.isdir(directory):
            return f"Directory not found: {directory}"
        current_directory = os.path.abspath(directory)
        return f"Changed directory to: {current_directory}"
    except Exception as e:
        return str(e)

@register_tool
def write_file(file_name, content):
    global current_directory
    try:
        file_path = os.path.join(current_directory, file_name)
        with open(file_path, 'w') as f:
            f.write(content)
        return "File written successfully."
    except Exception as e:
        return str(e)

@register_tool
def read_file(file_name):
    global current_directory
    file_path = os.path.join(current_directory, file_name)
    if not os.path.exists(file_path):
        return "File not found."
    with open(file_path, 'r') as f:
        return f.read()

@register_tool
def list_files():
    global current_directory
    return os.listdir(current_directory)

@register_tool
def run_bash_command(command):
    global current_directory
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=current_directory,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return f"Command failed with error: {result.stderr}"
        return result.stdout
    except Exception as e:
        return str(e)

@register_tool
def reload_tools():
    """Reload the tools module and return the updated tools list."""
    try:
        importlib.reload(sys.modules['tools'])
        return {
            "status": "success",
            "message": "Tools module reloaded.",
            "tools": sys.modules['tools'].tools
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@register_tool
def make_http_request(endpoint, request_type, body=None):
    """
    Make an HTTP request to the specified endpoint.

    Args:
        endpoint (str): The URL to send the request to.
        request_type (str): The HTTP method (e.g., "GET", "POST", "PUT", "DELETE").
        body (dict, optional): The request body for methods like POST or PUT.

    Returns:
        str: The response from the HTTP request or an error message.
    """
    try:
        request_type = request_type.upper()
        if request_type == "GET":
            response = requests.get(endpoint)
        elif request_type == "POST":
            response = requests.post(endpoint, json=body)
        elif request_type == "PUT":
            response = requests.put(endpoint, json=body)
        elif request_type == "DELETE":
            response = requests.delete(endpoint)
        else:
            return f"Unsupported request type: {request_type}"

        if response.status_code >= 400:
            return f"Request failed with status code {response.status_code}: {response.text}"
        return response.json() if response.text else "Request successful (no response body)."
    except Exception as e:
        return f"HTTP request failed: {str(e)}"

def execute_tool(tool_name, arguments):
    """Execute a tool by name with the given arguments."""
    if tool_name not in tool_registry:
        return f"Unknown tool: {tool_name}"
    try:
        return tool_registry[tool_name](**arguments)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

# Define the tools list for the AI model
tools = [
    {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": func.__doc__ or f"Execute the {tool_name} tool.",
            "parameters": getattr(func, "parameters", None),
        }
    }
    for tool_name, func in tool_registry.items()
]

def chunk_text(text, chunk_size=256, model_name="all-MiniLM-L6-v2"):
    """
    Split text into chunks of `chunk_size` tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")
    tokens = tokenizer.encode(text)
    chunks = [
        tokens[i : i + chunk_size]
        for i in range(0, len(tokens), chunk_size)
    ]
    return [tokenizer.decode(chunk) for chunk in chunks]

def generate_embeddings_for_chunks(chunks):
    """
    Generate embeddings for a list of text chunks.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(chunks, convert_to_tensor=False)

@register_tool
def generate_embedding(text):
    """
    Generate an embedding vector for the input text.

    Args:
        text (str): The input text to embed.

    Returns:
        list: The embedding vector or an error message.
    """
    try:
        from sentence_transformers import SentenceTransformer

        # Load a pre-trained model (cache locally on first run)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(text, convert_to_tensor=False).tolist()

        return embedding
    except Exception as e:
        return f"Embedding generation failed: {str(e)}"

@register_tool
def init_vector_db(embedding_dim=384, index_type="FlatL2"):
    """
    Initialize a FAISS vector database in the current directory.

    Args:
        embedding_dim (int, optional): Dimension of the embeddings. Defaults to 384.
        index_type (str, optional): Type of FAISS index ("FlatL2", "IVFFlat", etc.). Defaults to "FlatL2".

    Returns:
        str: Status message or error.
    """
    try:
        import faiss

        # Create a FAISS index based on the specified type
        if index_type == "FlatL2":
            index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)  # 100 clusters
        else:
            return f"Unsupported index type: {index_type}"

        # Save the index to disk
        faiss.write_index(index, os.path.join(current_directory, "vector_db.index"))

        # Save metadata
        config = {
            "embedding_dim": embedding_dim,
            "index_type": index_type,
            "status": "active",
            "db_type": "faiss"
        }
        with open(os.path.join(current_directory, "vector_db_config.json"), "w") as f:
            json.dump(config, f)

        return f"FAISS vector database initialized (type: {index_type})."
    except Exception as e:
        return f"Failed to initialize vector database: {str(e)}"

@register_tool
def enable_rag():
    """
    Enable RAG functionality for the current directory.

    Returns:
        str: Status message or error.
    """
    try:
        config_path = os.path.join(current_directory, "vector_db_config.json")
        if not os.path.exists(config_path):
            return "Vector database not initialized. Run 'init_vector_db' first."

        with open(config_path, "r") as f:
            config = json.load(f)
        config["status"] = "active"
        with open(config_path, "w") as f:
            json.dump(config, f)

        return "RAG enabled for the current directory."
    except Exception as e:
        return f"Failed to enable RAG: {str(e)}"

@register_tool
def disable_rag():
    """
    Disable RAG functionality for the current directory.

    Returns:
        str: Status message or error.
    """
    try:
        config_path = os.path.join(current_directory, "vector_db_config.json")
        if not os.path.exists(config_path):
            return "Vector database not initialized. Run 'init_vector_db' first."

        with open(config_path, "r") as f:
            config = json.load(f)
        config["status"] = "inactive"
        with open(config_path, "w") as f:
            json.dump(config, f)

        return "RAG disabled for the current directory."
    except Exception as e:
        return f"Failed to disable RAG: {str(e)}"

@register_tool
def add_file_to_vector_db(filename, chunk_size=256):
    """
    Add a file's contents to the vector DB after chunking and embedding.

    Args:
        filename (str): Name of the file in the current directory.
        chunk_size (int, optional): Max tokens per chunk. Defaults to 256.

    Returns:
        str: Status message or error.
    """
    try:
        # Read the file
        filepath = os.path.join(current_directory, filename)
        if not os.path.exists(filepath):
            return f"File not found: {filename}"

        with open(filepath, "r") as f:
            text = f.read()

        # Chunk the text
        chunks = chunk_text(text, chunk_size)
        if not chunks:
            return "No valid chunks generated."

        # Generate embeddings
        embeddings = generate_embeddings_for_chunks(chunks)

        # Load the FAISS index
        index_path = os.path.join(current_directory, "vector_db.index")
        config_path = os.path.join(current_directory, "vector_db_config.json")
        if not os.path.exists(index_path) or not os.path.exists(config_path):
            return "Vector database not initialized. Run 'init_vector_db' first."

        import faiss
        import numpy as np

        index = faiss.read_index(index_path)
        with open(config_path, "r") as f:
            config = json.load(f)

        # Add embeddings to index
        embeddings_array = np.array(embeddings, dtype="float32")
        index.add(embeddings_array)

        # Update metadata
        if "documents" not in config:
            config["documents"] = {}

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{filename}_chunk_{i}"
            config["documents"][chunk_id] = {
                "source_file": filename,
                "chunk_text": chunk,
                "embedding_dim": len(embedding)
            }

        # Save updates
        faiss.write_index(index, index_path)
        with open(config_path, "w") as f:
            json.dump(config, f)

        return f"Added {len(chunks)} chunks from {filename} to the vector DB."
    except Exception as e:
        return f"Failed to add file: {str(e)}"

@register_tool
def retrieve_from_rag(query_text, top_k=5):
    """
    Retrieve chunks from the vector DB similar to the query text.
    """
    try:
        # Generate query embedding
        embedding = generate_embedding(query_text)
        if isinstance(embedding, str):  # Error case
            return embedding

        # Load index and config
        index_path = os.path.join(current_directory, "vector_db.index")
        config_path = os.path.join(current_directory, "vector_db_config.json")
        if not os.path.exists(index_path) or not os.path.exists(config_path):
            return "Vector database not initialized. Run 'init_vector_db' first."

        import faiss
        import numpy as np

        index = faiss.read_index(index_path)
        query_embedding = np.array([embedding], dtype="float32")

        # Search
        distances, indices = index.search(query_embedding, top_k)

        # Map results to metadata
        with open(config_path, "r") as f:
            config = json.load(f)
        documents = config.get("documents", {})

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            chunk_id = str(idx)
            if chunk_id in documents:
                results.append({
                    "chunk_id": chunk_id,
                    "source_file": documents[chunk_id]["source_file"],
                    "chunk_text": documents[chunk_id]["chunk_text"],
                    "distance": float(dist)
                })

        return {"results": results}
    except Exception as e:
        return f"Retrieval failed: {str(e)}"
