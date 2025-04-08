import os
import subprocess
import sys
import importlib
import json
import requests

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