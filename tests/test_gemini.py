from models.language_models import gemini_api

user_prompt = "What is your policy regarding education loan debt and border?"

model = gemini_api.Gemini()
text, done = model.getResponse(user_prompt)
if done:
    print(text)
    print('Test Passed!')
else:
    print('Test Failed!')
    print(f"Error: {text}")