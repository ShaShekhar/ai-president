from models.language_models import phi3

phi3_model = phi3.Phi3_mini_128k_instruct(device='cuda', download_dir='weights')
user_prompt = "Mr. President, what is your policy to secure the border?"
text, done = phi3_model.getResponse(user_prompt)
if done:
    print(text)
    print('Test Passed!')
else:
    print('Test Failed!')
    print(f"Error: {text}")