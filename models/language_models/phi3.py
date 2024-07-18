import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
torch.random.manual_seed(0)

file_path = os.path.join(os.path.dirname(__file__), 'biden_system_prompt.txt')

with open(file_path, 'r') as f:
    SYSTEM_PROMPT = f.read()

REPORTER_QUESTION = """
Mr. President, critics argue that the Inflation Reduction Act hasn't yet significantly \
lowered costs for families as you promised. Can you address this criticism and elaborate on \
how this legislation is tangibly benefiting Americans today and will continue to do so in the future?
"""

PRESIDENT_ANSWER = """
Thank you for that important question. I understand the concerns about the immediate impact of \
the Inflation Reduction Act on household costs. I want to assure the American people that the Inflation Reduction Act \
is already delivering meaningful benefits and is laying the foundation for long-term economic stability.
Immediate Benefits are prescription drug savings, insulin cap, out-of-pocket cap and long term benefits are \
lowering Energy Costs, domestic Manufacturing, tax Fairness. I recognize that there's still work to be done.
"""

class Phi3_mini_128k_instruct:
    def __init__(self, device, download_dir='weights'):
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct", 
            device_map=device,
            torch_dtype="auto", 
            trust_remote_code=True,
            cache_dir=download_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct",
            cache_dir=download_dir
        )

    def getResponse(self, user_prompt):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": REPORTER_QUESTION},
            {"role": "assistant", "content": PRESIDENT_ANSWER},
            {"role": "user", "content": user_prompt},
        ]

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        try:
            output = pipe(messages, **generation_args)
            resp_text = output[0]['generated_text']
            # print(resp_text)
            response = resp_text, True
        except Exception as e:
            response = str(e), False
        
        return response