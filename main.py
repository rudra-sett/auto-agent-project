from openai import OpenAI
from tools import execute_tool, tools
import os
import json
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize the client
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

system = '''
You are an AI assistant that can read, write, and list files. 
You can use these capabilities to change your own abilities and system prompt as requested by the user. 
Otherwise, you will just act as a normal, knowledgeable assistant. Whenever you use a tool, please also say what you are doing.
'''

messages = [{"role": "system", "content": system}]

def run():
    done = False
    while not done:
        try:
            prompt = input("You: ")
            if prompt == "exit":
                done = True
                break
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools,
                stream=False
            )

            if response.choices[0].message.tool_calls:
                for tool in response.choices[0].message.tool_calls:
                    tool_name = tool.function.name
                    tool_args = json.loads(tool.function.arguments)
                    result = execute_tool(tool_name, tool_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "content": str(result)
                    })
            else:
                print("AI:", response.choices[0].message.content)

        except Exception as e:
            logging.error(f"Error: {e}")
            done = True

if __name__ == "__main__":
    run()