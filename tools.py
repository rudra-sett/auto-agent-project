import os
import subprocess
import sys
import importlib

def write_file(file_name, content):
    try:
      with open(file_name, 'w') as f:
        f.write(content)
    except Exception as e:
      return str(e)
    return "File written successfully."

def read_file(file_name):
    if not os.path.exists(file_name):
        return "File not found."
    with open(file_name, 'r') as f:
        return f.read()

def list_files():
    return os.listdir()

def run_bash_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Command failed with error: {result.stderr}"
        return result.stdout
    except Exception as e:
        return str(e)

def restart():
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
        return "Restarting..."
    except Exception as e:
        return str(e)

def switch_tools(version):
    try:
        global tools
        tools_module = importlib.import_module(f"tools_{version}")
        tools = tools_module.tools
        return f"Switched to tools version {version}"
    except Exception as e:
        return str(e)

tools = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file with a file name and content",
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Name of the file."
                },
                "content": {
                    "type": "string",
                    "description": "Content you want to write to the file."
                }
            },
            "required": [
                "file_name",
                "content"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file and return the content",
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Name of the file."
                }
            },
            "required": [
                "file_name"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List all files in the current directory",
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "run_bash_command",
        "description": "Run a bash command on the system",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run."
                }
            },
            "required": [
                "command"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "restart",
        "description": "Restart the AI assistant to apply changes",
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "switch_tools",
        "description": "Switch between different versions of tools",
        "parameters": {
            "type": "object",
            "properties": {
                "version": {
                    "type": "string",
                    "description": "The version of tools to switch to (e.g., 'v1', 'v2')."
                }
            },
            "required": [
                "version"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}
]