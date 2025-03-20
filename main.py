from openai import OpenAI
from tools import *
import os
import json
import logging
import importlib

# load environment variables
from dotenv import load_dotenv
load_dotenv()

# get key from environment variable
key = os.getenv("DEEPSEEK_API_KEY")
if key is None:
  raise ValueError("DeepSeek API Key not found in environment variables")

client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

system = '''
You are an AI assistant that can read, write, and list files. 
You can use these capabilities to change your own abilities and system prompt as requested by the user. 
Otherwise, you will just act as a normal, knowledgeable assistant. Whenever you use a tool, please also say what you are doing.

You are currently using version v1 of the tools module.
'''

messages = [
    {
        "role": "system",
        "content": system
    }
]

def switch_tools(version):
    try:
        global tools
        tools_module = importlib.import_module(f"tools_{version}")
        tools = tools_module.tools
        messages[0]['content'] = f'''
You are an AI assistant that can read, write, and list files. 
You can use these capabilities to change your own abilities and system prompt as requested by the user. 
Otherwise, you will just act as a normal, knowledgeable assistant. Whenever you use a tool, please also say what you are doing.

You are currently using version {version} of the tools module.
'''
        return f"Switched to tools version {version}"
    except Exception as e:
        return str(e)

def run():
    done = False
    skip = False
    while not done:
      try:
        if not skip:
          prompt = input("You: ")
          if prompt == "exit":
            done = True
            break
          messages.append({
              "role": "user",
              "content": prompt
          })

        response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        stream=False
        )

        messages.append(response.choices[0].message)
        if response.choices[0].message.tool_calls is not None:
          for tool in response.choices[0].message.tool_calls:
            print(tool.function.name)
            args = json.loads(tool.function.arguments)
            print(args)
            if tool.function.name == "write_file":
              result = write_file(**args)
              messages.append({
                  "role": "tool",
                  "tool_call_id": tool.id,
                  "content": result
              })
            elif tool.function.name == "read_file":
              content = read_file(**args)
              messages.append({
                  "role": "tool",
                  "tool_call_id": tool.id,
                  "content": content
              })
            elif tool.function.name == "list_files":
              files = list_files()
              messages.append({
                  "role": "tool",
                  "tool_call_id": tool.id,
                  "content": json.dumps(files)
              })
            elif tool.function.name == "run_bash_command":
              result = run_bash_command(**args)
              messages.append({
                  "role": "tool",
                  "tool_call_id": tool.id,
                  "content": result
              })            
            elif tool.function.name == "switch_tools":
              result = switch_tools(**args)
              messages.append({
                  "role": "tool",
                  "tool_call_id": tool.id,
                  "content": result
              })
            else:
              print("Unknown tool used: " + tool.tool)
          skip = True 
        else:
          skip = False

        print("AI: " + response.choices[0].message.content)
      except Exception as e:
        logging.error(f"An error occurred: {e}")
        done = True

if __name__ == "__main__":
    run()