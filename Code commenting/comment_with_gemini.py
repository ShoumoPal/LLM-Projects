import google.generativeai as genai
from load_dotenv import load_dotenv
from typing import List
import argparse

# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI API
genai.configure()

# System prompt for the Gemini model
system_prompt = "You are a helpful assistant. That is very good at writing comments for code. You will be given a code snippet and you will add comments to the code snippet. Do not change the code, just add comments to it."

def user_prompt_for(code):
    # Construct the user prompt with instructions and the code snippet
    user_prompt = "Add apt comments to the following code."
    user_prompt += "Do not add any other unnecessary text or details, just the unmodified code with comments.\n\n"
    user_prompt += code
    return user_prompt

def get_arg_parser():
    # Create an argument parser to handle command line arguments
    parser = argparse.ArgumentParser(description="Comment code using Google Generative AI.")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()
    return args

def generate_comment_file(input_file: str, output_file: str):
    # Read the code from the input file
    with open(input_file, "r") as file:
        code = file.read()

    # Generate comments using the Gemini model
    response = model.generate_content(user_prompt_for(code))

    # Write the commented code to the output file, removing ```python``` if present
    with open(output_file, "w") as file:
        output = response.text.strip()
        if output.startswith("```python"):
            output = output[9:]
        if output.endswith("```"):
            output = output[:-3]
        file.write(output.strip())

# Initialize the Gemini model with the system prompt
model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_prompt)

if __name__ == "__main__":
    # Parse command line arguments and generate the commented code file
    args = get_arg_parser()
    generate_comment_file(args.input_file, args.output_file)