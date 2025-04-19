\
from openai import OpenAI
from tools import execute_tool, tools
import os
import json
import logging
from dotenv import load_dotenv

# --- History Configuration ---
HISTORY_FILE = "conversation_history.jsonl"
MESSAGES_TO_LOAD = 10
# --------------------------

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Client Initialization ---
# Determine which model to use based on .env
model = os.getenv("MODEL", "DeepSeek")  # Default to DeepSeek if MODEL is not set
logging.info(f"Using model: {model}")

if model == "DeepSeek":
    # Initialize DeepSeek client
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    model_name = "deepseek-chat"
elif model == "Gemini":
    # Initialize Gemini client (replace with actual Gemini initialization)
    client = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")  # Placeholder
    model_name = "gemini-2.0-flash" # Replace with actual Gemini model if needed
else:
    logging.error(f"Unsupported model specified in .env: {model}")
    raise ValueError(f"Unsupported model: {model}")
# ---------------------------

system = '''
You are an AI assistant that can read, write, and list files.
You can use these capabilities to change your own abilities and system prompt as requested by the user.
Otherwise, you will just act as a normal, knowledgeable assistant. Whenever you use a tool, please also say what you are doing.
'''

# --- History Loading Function ---
def load_history(filename, num_messages):
    """Loads the last num_messages from a JSON Lines file."""
    history = []
    if not os.path.exists(filename):
        logging.info(f"History file '{filename}' not found. Starting fresh.")
        return history

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Get the last 'num_messages' lines
        start_index = max(0, len(lines) - num_messages)
        relevant_lines = lines[start_index:]

        for i, line in enumerate(relevant_lines):
            line_num = start_index + i + 1
            try:
                if line.strip(): # Ensure line is not empty
                    history.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping corrupted line {line_num} in '{filename}': {e} - Line content: '{line.strip()}'")
        logging.info(f"Loaded {len(history)} messages from '{filename}'.")
        return history
    except IOError as e:
        logging.error(f"Error reading history file '{filename}': {e}")
        return [] # Return empty list on error
    except Exception as e:
        logging.error(f"Unexpected error loading history from '{filename}': {e}")
        return []

# --- History Saving Function ---
def save_message(filename, message_dict):
    """Appends a message dictionary to a JSON Lines file."""
    try:
        # Ensure message_dict is serializable (handle potential non-dict objects if needed)
        if not isinstance(message_dict, dict):
             # Attempt conversion for common OpenAI response objects
            if hasattr(message_dict, 'model_dump') and callable(message_dict.model_dump):
                 message_dict = message_dict.model_dump(exclude_unset=True) # Use model_dump for pydantic models
            else:
                 # Basic fallback (might need adjustment based on actual object structure)
                 logging.warning(f"Message is not a dict, attempting basic conversion: {type(message_dict)}")
                 message_dict = {
                     "role": getattr(message_dict, 'role', 'unknown'),
                     "content": getattr(message_dict, 'content', str(message_dict))
                 }
                 # Add tool_calls if present
                 if hasattr(message_dict, 'tool_calls') and message_dict.tool_calls:
                     # Ensure tool_calls are also serializable
                     serializable_tool_calls = []
                     for tc in message_dict.tool_calls:
                         if hasattr(tc, 'model_dump') and callable(tc.model_dump):
                             serializable_tool_calls.append(tc.model_dump(exclude_unset=True))
                         else:
                             # Basic fallback for tool calls
                              serializable_tool_calls.append({
                                  "id": getattr(tc, 'id', None),
                                  "type": getattr(tc, 'type', 'function'),
                                  "function": {
                                      "name": getattr(getattr(tc, 'function', None), 'name', None),
                                      "arguments": getattr(getattr(tc, 'function', None), 'arguments', None)
                                   }
                              })
                     message_dict["tool_calls"] = serializable_tool_calls


        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(message_dict) + '\n')
    except IOError as e:
        logging.error(f"Error writing to history file '{filename}': {e}")
    except TypeError as e:
        logging.error(f"Error serializing message to JSON: {e} - Message: {message_dict}")
    except Exception as e:
        logging.error(f"Unexpected error saving message to '{filename}': {e}")

# --- Initialize Messages ---
# Start with the system prompt
messages = [{"role": "system", "content": system}]

# Load historical messages
historical_messages = load_history(HISTORY_FILE, MESSAGES_TO_LOAD)
messages.extend(historical_messages)
logging.info(f"Total messages after loading history: {len(messages)}")
# --------------------------


def run():
    done = False
    skip_prompt = False # Renamed 'skip' to 'skip_prompt' for clarity

    while not done:
        try:
            # 1. Get User Input (or skip if processing tool results)
            if not skip_prompt:
                prompt = input("You: ")
                if prompt.lower() == "exit":
                    done = True
                    logging.info("Exiting application.")
                    break

                user_message = {"role": "user", "content": prompt}
                messages.append(user_message)
                save_message(HISTORY_FILE, user_message) # Save user message

            # Reset skip flag
            skip_prompt = False

            # 2. Get Model Response
            # logging.info("Sending request to AI model...")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                stream=False # Assuming stream=False based on original code
            )

            assistant_message_obj = response.choices[0].message

            # Prepare the assistant message dictionary for appending and saving
            assistant_message_dict = None
            if hasattr(assistant_message_obj, 'model_dump') and callable(assistant_message_obj.model_dump):
                 assistant_message_dict = assistant_message_obj.model_dump(exclude_unset=True)
            else: # Basic fallback
                 assistant_message_dict = {
                     "role": assistant_message_obj.role,
                     "content": assistant_message_obj.content
                 }
                 if assistant_message_obj.tool_calls:
                     # Basic serialization for tool calls if model_dump isn't available
                     assistant_message_dict["tool_calls"] = [
                         {
                             "id": tc.id,
                             "type": tc.type,
                             "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                         } for tc in assistant_message_obj.tool_calls
                    ]

            messages.append(assistant_message_dict) # Append the dictionary
            save_message(HISTORY_FILE, assistant_message_dict) # Save assistant message
            # logging.info("Assistant message saved.")

            # 3. Handle Tool Calls (if any)
            if assistant_message_obj.tool_calls:
                logging.info(f"Detected tool calls: {[tc.function.name for tc in assistant_message_obj.tool_calls]}")
                # Print assistant's reasoning before executing tools (if any content exists)
                if assistant_message_obj.content:
                    print("AI:", assistant_message_obj.content)

                for tool_call in assistant_message_obj.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding arguments for tool '{tool_name}': {e}. Arguments: {tool_call.function.arguments}")
                        tool_result = f"Error: Invalid JSON arguments provided for tool {tool_name}."
                    else:
                        # logging.info(f"Executing tool: {tool_name} with args: {tool_args}")
                        tool_result = execute_tool(tool_name, tool_args)
                        # logging.info(f"Tool result: {tool_result}")


                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name, # Include tool name for clarity, matching some API expectations
                        "content": str(tool_result) # Ensure content is string
                    }
                    messages.append(tool_message)
                    save_message(HISTORY_FILE, tool_message) # Save tool message
                    # logging.info(f"Tool message for '{tool_name}' saved.")

                # Set flag to skip user prompt and immediately call model again with tool results
                skip_prompt = True

            # 4. Print Assistant's Final Response (if no tool calls or after tool calls if content exists)
            else:
                skip_prompt = False # Ensure we get user input next time
                if assistant_message_obj.content:
                    print("AI:", assistant_message_obj.content)
                else:
                    # Handle cases where the assistant might not return content (e.g., only tool calls requested but none found?)
                    logging.warning("Assistant message had no content and no tool calls.")
                    print("AI: (No text response)")


        except Exception as e:
            logging.exception(f"An unexpected error occurred in the main loop: {e}")
            # Option to continue or exit on error
            # done = True # Uncomment to exit on any error
            print(f"An error occurred: {e}. Please try again or type 'exit'.")
            skip_prompt = False # Ensure we ask for input again after an error

if __name__ == "__main__":
    run()

