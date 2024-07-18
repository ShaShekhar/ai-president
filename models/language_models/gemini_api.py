import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import os, sys

qa = """
REPORTER: Mr. President, critics argue that the Inflation Reduction Act hasn't yet significantly \
lowered costs for families as you promised. Can you address this criticism and elaborate on \
how this legislation is tangibly benefiting Americans today and will continue to do so in the future?

PRESIDENT: Thank you for that important question. I understand the concerns about the immediate impact of \
the Inflation Reduction Act on household costs. I want to assure the American people that the Inflation Reduction Act \
is already delivering meaningful benefits and is laying the foundation for long-term economic stability.
Immediate Benefits are prescription drug savings, insulin cap, out-of-pocket cap and long term benefits are \
lowering Energy Costs, domestic Manufacturing, tax Fairness. I recognize that there's still work to be done.
"""

class Gemini:
    def __init__(self, api_key=None, model_name="gemini-pro"):
        if api_key is None:
            load_dotenv(find_dotenv())
            api_key=os.getenv("GEMINI_API_KEY")

        genai.configure(api_key=api_key)
        
        if not api_key:
            raise ValueError(
                "GEMINI API key not found. "
                "Please set the GEMINI_API_KEY environment variable."
            )
        self.model = genai.GenerativeModel(model_name)

        file_path = os.path.join(os.path.dirname(__file__), 'biden_system_prompt.txt')
        
        with open(file_path, 'r') as f:
            system_prompt = f.read()

        self.prompt = system_prompt + qa
    
    def list_model(self):
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)

    def getResponse(self, user_prompt):
        
        query = "\nREPORTER: {} \n\nPRESIDENT:".format(user_prompt)
        final_query = self.prompt + query
        
        try:
            response = self.model.generate_content(final_query)
            resp_text = response.text
            # print(resp_text)
            response = resp_text, True
        except Exception as e:
            response = str(e), False
        
        return response