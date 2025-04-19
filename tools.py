# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import importlib
import json
import requests
import logging

# --- RAG Dependencies ---
# Wrap in try-except to allow other tools to function if RAG components are missing
try:
    import faiss
    import numpy as np
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    RAG_ENABLED = True
    # Load the embedding model globally once for efficiency
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Load tokenizer globally
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

except ImportError as e:
    logging.warning(f"RAG dependencies not found, RAG tools will be unavailable: {e}")
    RAG_ENABLED = False
    faiss = None
    np = None
    AutoTokenizer = None
    SentenceTransformer = None
    embedding_model = None
    tokenizer = None
# --- End RAG Dependencies ---


# Global variable to track the current working directory
current_directory = os.getcwd()

# Registry mapping tool names to functions
tool_registry = {}

def register_tool(func):
    """Decorator to register a tool in the registry."""
    # Only register RAG tools if dependencies are met
    if func.__name__.startswith("rag_") and not RAG_ENABLED:
        logging.info(f"Skipping registration of RAG tool '{func.__name__}' due to missing dependencies.")
        return func # Return the function itself, but don't add to registry

    # Add non-RAG tools or RAG tools if dependencies are met
    tool_registry[func.__name__] = func
    return func

# --- Standard File/System Tools ---

@register_tool
def change_directory(directory):
    """Change the current working directory.
    Args:
        directory (str): The target directory path.
    Returns:
        str: Confirmation message or error.
    """
    global current_directory
    try:
        new_path = os.path.abspath(os.path.join(current_directory, directory))
        if not os.path.isdir(new_path):
            return f"Error: Directory not found: {new_path}"
        current_directory = new_path
        return f"Changed directory to: {current_directory}"
    except Exception as e:
        return f"Error changing directory: {str(e)}"

@register_tool
def write_file(file_name, content):
    """Write content to a file in the current directory.
    Args:
        file_name (str): The name of the file to write.
        content (str): The content to write into the file.
    Returns:
        str: Confirmation message or error.
    """
    global current_directory
    try:
        file_path = os.path.join(current_directory, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"File '{file_name}' written successfully in {current_directory}."
    except Exception as e:
        return f"Error writing file: {str(e)}"

@register_tool
def read_file(file_name):
    """Read the content of a file in the current directory.
    Args:
        file_name (str): The name of the file to read.
    Returns:
        str: The content of the file or an error message.
    """
    global current_directory
    file_path = os.path.join(current_directory, file_name)
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_name}"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@register_tool
def list_files(path="."):
    """List files and directories in the specified path relative to the current directory.
    Args:
        path (str, optional): The relative path to list. Defaults to the current directory (".").
    Returns:
        list: A list of file and directory names or an error message string.
    """
    global current_directory
    target_path = os.path.abspath(os.path.join(current_directory, path))
    if not os.path.isdir(target_path):
        return f"Error: Path not found or not a directory: {target_path}"
    try:
        return os.listdir(target_path)
    except Exception as e:
        return f"Error listing files: {str(e)}"


@register_tool
def run_bash_command(command):
    """Run a bash command in the current directory.
    Args:
        command (str): The bash command to run.
    Returns:
        str: The output (stdout and stderr) of the command or an error message.
    """
    global current_directory
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=current_directory,
            capture_output=True,
            text=True,
            check=False # Don't raise exception on non-zero exit
        )
        output = f"Exit Code: {result.returncode}"

        if result.stdout:
            output += f"Stdout: {result.stdout}"
        if result.stderr:
            output += f"Stderr:{result.stderr}"
        return output.strip()

    except Exception as e:
        return f"Error running bash command: {str(e)}"

@register_tool
def reload_tools():
    """Reload the tools module and return the updated tools list."""
    global tool_registry, tools # Need to update global 'tools' list as well
    try:
        # Clear the existing registry
        current_registry_keys = list(tool_registry.keys())
        for key in current_registry_keys:
             del tool_registry[key]

        # Reload the module
        importlib.reload(sys.modules['tools'])

        # Re-populate the registry and tools list from the reloaded module
        # This assumes the reloaded module follows the same registration pattern
        # and defines a 'tools' list at the module level.
        reloaded_module = sys.modules['tools']
        tool_registry.update(getattr(reloaded_module, 'tool_registry', {}))
        new_tools_list = getattr(reloaded_module, 'tools', [])

        # Update the global tools variable
        globals()['tools'] = new_tools_list

        return {
            "status": "success",
            "message": f"Tools module reloaded. {len(tool_registry)} tools registered.",
            "registered_tools": list(tool_registry.keys()),
            "api_schema_tools_count": len(new_tools_list)
        }
    except Exception as e:
        # Attempt to restore previous state if reload fails
        # Note: This is complex and might not fully restore state.
        return {"status": "error", "message": f"Failed to reload tools: {str(e)}"}


@register_tool
def make_http_request(endpoint, request_type, body=None):
    """Make an HTTP request to the specified endpoint.
    Args:
        endpoint (str): The URL to send the request to.
        request_type (str): The HTTP method (e.g., "GET", "POST", "PUT", "DELETE").
        body (dict, optional): The request body for methods like POST or PUT. Defaults to None.
    Returns:
        dict or str: The JSON response from the HTTP request or an error message string.
    """
    try:
        request_type = request_type.upper()
        headers = {'Content-Type': 'application/json'}
        response = None

        if request_type == "GET":
            response = requests.get(endpoint, headers=headers)
        elif request_type == "POST":
            response = requests.post(endpoint, json=body, headers=headers)
        elif request_type == "PUT":
            response = requests.put(endpoint, json=body, headers=headers)
        elif request_type == "DELETE":
            response = requests.delete(endpoint, headers=headers)
        else:
            return f"Error: Unsupported request type: {request_type}"

        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Try to parse JSON, otherwise return text
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text if response.text else "Request successful (no response body)."

    except requests.exceptions.RequestException as e:
        return f"Error: HTTP request failed: {str(e)}"
    except Exception as e:
         return f"Error during HTTP request: {str(e)}"

# --- RAG Helper Functions ---

def _load_rag_config():
    """Loads the RAG configuration file."""
    config_path = os.path.join(current_directory, "vector_db_config.json")
    if not os.path.exists(config_path):
        return None, "Error: RAG not initialized in this directory. Run 'rag_init_vector_db' first."
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config, None
    except Exception as e:
        return None, f"Error loading RAG config: {str(e)}"

def _save_rag_config(config):
    """Saves the RAG configuration file."""
    config_path = os.path.join(current_directory, "vector_db_config.json")
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        return None
    except Exception as e:
        return f"Error saving RAG config: {str(e)}"

def _load_faiss_index(config):
    """Loads the FAISS index based on config."""
    index_path = os.path.join(current_directory, config.get("index_file", "vector_db.faiss"))
    if not os.path.exists(index_path):
        return None, f"Error: FAISS index file not found: {index_path}"
    try:
        index = faiss.read_index(index_path)
        return index, None
    except Exception as e:
        return None, f"Error loading FAISS index: {str(e)}"

def _save_faiss_index(index, config):
    """Saves the FAISS index."""
    index_path = os.path.join(current_directory, config.get("index_file", "vector_db.faiss"))
    try:
        faiss.write_index(index, index_path)
        return None
    except Exception as e:
        return f"Error saving FAISS index: {str(e)}"

def _chunk_text(text, chunk_size=256, chunk_overlap=32):
    """Split text into potentially overlapping chunks based on token count."""
    if not RAG_ENABLED or not tokenizer:
         return ["Error: RAG components (tokenizer) not available."], None
    try:
        tokens = tokenizer.encode(text)
        if not tokens:
            return [], "Input text resulted in no tokens."

        chunks = []
        start_index = 0
        while start_index < len(tokens):
            end_index = min(start_index + chunk_size, len(tokens))
            chunk_tokens = tokens[start_index:end_index]
            chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))

            # Move start index for the next chunk
            start_index += chunk_size - chunk_overlap
            # If overlap is large, ensure we don't get stuck
            if start_index + chunk_size <= end_index :
                 start_index = end_index # Prevent infinite loops with large overlaps

        return chunks, None
    except Exception as e:
        return [], f"Error chunking text: {str(e)}"


def _generate_embeddings(text_chunks):
    """Generate embeddings for a list of text chunks."""
    if not RAG_ENABLED or not embedding_model:
        return None, "Error: RAG components (embedding model) not available."
    try:
        embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
        # Ensure embeddings are float32 numpy arrays for FAISS
        return np.array(embeddings, dtype="float32"), None
    except Exception as e:
        return None, f"Error generating embeddings: {str(e)}"


# --- RAG Management Tools ---

@register_tool
def rag_init_vector_db(embedding_dim=384, index_type="FlatL2"):
    """Initialize a FAISS vector database and RAG config in the current directory.
       Currently only supports 'FlatL2' index type.

    Args:
        embedding_dim (int, optional): Dimension of the embeddings (default 384 for all-MiniLM-L6-v2).
        index_type (str, optional): Type of FAISS index. Currently must be "FlatL2". Defaults to "FlatL2".

    Returns:
        str: Status message or error.
    """
    if not RAG_ENABLED:
        return "Error: RAG dependencies (faiss, sentence-transformers) are not installed."

    global current_directory
    config_path = os.path.join(current_directory, "vector_db_config.json")
    index_file = "vector_db.faiss"
    index_path = os.path.join(current_directory, index_file)

    if os.path.exists(config_path) or os.path.exists(index_path):
        return "Error: RAG vector database or config already seems to exist in this directory."

    if index_type != "FlatL2":
        return f"Error: Currently only 'FlatL2' index type is supported for automatic initialization."

    try:
        # Create FAISS index
        index = faiss.IndexFlatL2(embedding_dim)

        # Save the empty index
        faiss.write_index(index, index_path)

        # Create and save metadata/config
        config = {
            "embedding_dim": embedding_dim,
            "embedding_model": "all-MiniLM-L6-v2", # Store which model was used
            "index_type": index_type,
            "index_file": index_file,
            "status": "active", # 'active' or 'inactive'
            "chunk_size": 256, # Default chunk size
            "chunk_overlap": 32, # Default overlap
            "metadata_store": [] # List to store metadata, index matches FAISS index
        }
        error = _save_rag_config(config)
        if error:
            # Clean up index file if config saving failed
            if os.path.exists(index_path): os.remove(index_path)
            return error

        return f"FAISS vector database ('{index_file}') and config initialized successfully (Type: {index_type}, Dim: {embedding_dim}). RAG is active."

    except Exception as e:
        # Clean up potentially partially created files
        if os.path.exists(index_path): os.remove(index_path)
        if os.path.exists(config_path): os.remove(config_path)
        return f"Error initializing vector database: {str(e)}"

@register_tool
def rag_set_status(status):
    """Enable ('active') or disable ('inactive') RAG functionality for the current directory.

    Args:
        status (str): The desired status ('active' or 'inactive').

    Returns:
        str: Status message or error.
    """
    if not RAG_ENABLED: return "Error: RAG dependencies not available."
    config, error = _load_rag_config()
    if error: return error

    status = status.lower()
    if status not in ["active", "inactive"]:
        return "Error: Status must be 'active' or 'inactive'."

    config["status"] = status
    error = _save_rag_config(config)
    if error: return error

    return f"RAG status set to '{status}' for the current directory."

@register_tool
def rag_get_status():
    """Check the current status and configuration of RAG for this directory.

    Returns:
        dict or str: RAG configuration details or an error message.
    """
    if not RAG_ENABLED: return "Error: RAG dependencies not available."
    config, error = _load_rag_config()
    if error: return error

    index, error = _load_faiss_index(config)
    if error:
        # Provide config info even if index loading fails
         return {"config_status": config, "index_status": error}

    status_info = {
        "rag_status": config.get("status", "unknown"),
        "embedding_model": config.get("embedding_model", "unknown"),
        "embedding_dimension": config.get("embedding_dim", "unknown"),
        "index_type": config.get("index_type", "unknown"),
        "index_file": config.get("index_file", "unknown"),
        "index_vector_count": index.ntotal if index else "N/A",
        "metadata_chunk_count": len(config.get("metadata_store", [])),
        "chunk_size": config.get("chunk_size", "default"),
        "chunk_overlap": config.get("chunk_overlap", "default"),
        "config_file_path": os.path.join(current_directory, "vector_db_config.json"),
        "index_file_path": os.path.join(current_directory, config.get("index_file", "vector_db.faiss"))
    }
    # Check for potential mismatch
    if index and index.ntotal != len(config.get("metadata_store", [])):
        status_info["warning"] = "Warning: FAISS index count mismatches metadata store count!"

    return status_info


# --- RAG Data Handling Tools ---

@register_tool
def rag_add_file(file_name):
    """Chunk, embed, and add a file's content to the RAG vector database.

    Args:
        file_name (str): Name of the text file in the current directory to add.

    Returns:
        str: Status message indicating success or failure, including number of chunks added.
    """
    if not RAG_ENABLED: return "Error: RAG dependencies not available."
    global current_directory

    # 1. Load Config and check status
    config, error = _load_rag_config()
    if error: return error
    if config.get("status") != "active":
        return "Error: RAG is currently inactive for this directory. Use 'rag_set_status' to activate."

    # 2. Read the file
    file_content = read_file(file_name)
    if file_content.startswith("Error:"):
        return file_content # Propagate read_file error

    # 3. Load the index
    index, error = _load_faiss_index(config)
    if error: return error

    # 4. Chunk the text
    chunk_size = config.get("chunk_size", 256)
    chunk_overlap = config.get("chunk_overlap", 32)
    chunks, error = _chunk_text(file_content, chunk_size, chunk_overlap)
    if error: return f"Error processing file '{file_name}': {error}"
    if not chunks: return f"No text chunks were generated from '{file_name}'."

    # 5. Generate embeddings
    embeddings, error = _generate_embeddings(chunks)
    if error: return f"Error generating embeddings for '{file_name}': {error}"
    if embeddings.shape[0] != len(chunks):
         return f"Error: Mismatch between number of chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) for '{file_name}'."
    if embeddings.shape[1] != index.d:
        return f"Error: Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({index.d}). Initialize DB with correct dimension."


    # 6. Add embeddings to index
    try:
        index.add(embeddings)
    except Exception as e:
        return f"Error adding embeddings to FAISS index for '{file_name}': {str(e)}"

    # 7. Add corresponding metadata to config store (CRITICAL: must match embedding order)
    metadata_store = config.get("metadata_store", [])
    for i, chunk_text in enumerate(chunks):
        metadata_store.append({
            "source_file": file_name,
            "chunk_text": chunk_text,
            "original_chunk_index": i # Optional: track index within the file
        })
    config["metadata_store"] = metadata_store # Update config dict

    # 8. Save updated index and config
    error = _save_faiss_index(index, config)
    if error: return f"Error saving updated FAISS index for '{file_name}': {error}. Metadata may be out of sync."
    error = _save_rag_config(config)
    if error: return f"Error saving updated config for '{file_name}': {error}. Index may be out of sync."

    return f"Successfully added {len(chunks)} chunks from '{file_name}' to the RAG database. Total vectors: {index.ntotal}."


@register_tool
def rag_retrieve(query_text, top_k=5):
    """Retrieve relevant text chunks from the RAG database based on a query.

    Args:
        query_text (str): The query text to search for.
        top_k (int, optional): The number of top results to retrieve. Defaults to 5.

    Returns:
        dict: A dictionary containing the list of results or an error message.
              Each result includes 'source_file', 'chunk_text', 'distance', and 'vector_id'.
    """
    if not RAG_ENABLED: return {"error": "RAG dependencies not available."}

    # 1. Load config and check status
    config, error = _load_rag_config()
    if error: return {"error": error}
    if config.get("status") != "active":
        return {"error": "RAG is currently inactive for this directory."}

    # 2. Load index and metadata store
    index, error = _load_faiss_index(config)
    if error: return {"error": error}
    metadata_store = config.get("metadata_store", [])
    if not metadata_store:
        return {"results": [], "message": "The RAG database is empty."}

    # 3. Check for mismatch (important sanity check)
    if index.ntotal != len(metadata_store):
         return {"error": f"Error: FAISS index count ({index.ntotal}) mismatches metadata store count ({len(metadata_store)}). Database may be corrupt."}


    # 4. Generate query embedding
    query_embedding, error = _generate_embeddings([query_text]) # Pass as list
    if error: return {"error": f"Failed to generate embedding for query: {error}"}
    if query_embedding.shape[1] != index.d:
         return {"error": f"Query embedding dimension ({query_embedding.shape[1]}) does not match index dimension ({index.d})."}


    # 5. Perform search
    try:
        distances, indices = index.search(query_embedding, top_k)
    except Exception as e:
        return {"error": f"Error during FAISS search: {str(e)}"}

    # 6. Format results using direct index lookup
    results = []
    if indices.size > 0: # Check if any results were found
        for i in range(indices.shape[1]): # Iterate through columns (top_k results)
            vector_id = int(indices[0, i]) # Get the FAISS vector ID (the direct index)
            distance = float(distances[0, i])

            # Check if the index is valid for our metadata list
            if 0 <= vector_id < len(metadata_store):
                metadata = metadata_store[vector_id]
                results.append({
                    "vector_id": vector_id,
                    "source_file": metadata.get("source_file", "Unknown"),
                    "chunk_text": metadata.get("chunk_text", "N/A"),
                    "distance": distance # Lower distance usually means more similar for L2
                })
            else:
                # This shouldn't happen if counts match, but good to handle
                logging.warning(f"Retrieved invalid index {vector_id} from FAISS search.")
                results.append({
                    "vector_id": vector_id,
                    "error": "Retrieved index out of bounds for metadata store.",
                    "distance": distance
                })

    return {"results": results}


# --- Tool Execution Logic ---

def execute_tool(tool_name, arguments):
    """Execute a tool by name with the given arguments."""
    if tool_name not in tool_registry:
        # Check if it's an inactive RAG tool
        if tool_name.startswith("rag_") and not RAG_ENABLED:
             return f"Error: Tool '{tool_name}' requires RAG dependencies (faiss, sentence-transformers, numpy, transformers) which are not installed or failed to import."
        return f"Error: Unknown tool: {tool_name}"

    func = tool_registry[tool_name]
    try:
        # Simple argument handling: pass the dict directly
        # Assumes API call maps arguments correctly to function parameters
        return func(**arguments)
    except TypeError as e:
         return f"Error executing tool '{tool_name}': Incorrect arguments provided. {str(e)}"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


# Define the tools list for the AI model's API schema
# Dynamically generate based on registered tools

tools = []
for tool_name, func in tool_registry.items():
    # Basic parameter extraction (can be enhanced)
    import inspect
    sig = inspect.signature(func)
    params = sig.parameters
    param_details = {}
    required_params = []
    for name, param in params.items():
        param_details[name] = {
            "type": "string" if param.annotation == inspect.Parameter.empty else str(param.annotation),
            "description": f"{name} parameter" # Placeholder description
        }
        if param.default == inspect.Parameter.empty:
            required_params.append(name)

    # Simple check for optional dictionary for body
    if tool_name == 'make_http_request' and 'body' in param_details:
        param_details['body']['type'] = 'object' # Mark as object for JSON body
        param_details['body']['description'] = 'The request body for methods like POST or PUT (JSON object).'


    tools.append({
        "type": "function",
        "function": {
            "name": tool_name,
            "description": inspect.getdoc(func) or f"Execute the {tool_name} tool.",
            "parameters": {
                "type": "object",
                "properties": param_details,
                "required": required_params
            },
        }
    })

# Example of how to potentially add descriptions manually if needed:
# for tool_def in tools:
#     if tool_def["function"]["name"] == "write_file":
#         tool_def["function"]["parameters"]["properties"]["file_name"]["description"] = "The name of the file to create or overwrite."
#         tool_def["function"]["parameters"]["properties"]["content"]["description"] = "The text content to write into the file."

