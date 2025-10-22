from openai import OpenAI
import random

API_KEY = "" 
with open("../api_key.txt") as f:
    API_KEY = f.read()

client = OpenAI(api_key = API_KEY)

def get_output(prompt, temperature):
    output = client.chat.completions.create(
    model = "gpt-4o",
    temperature = temperature,
    messages = [{"role":"user", "content": prompt}]
    )
    return output

###  Question 1-2

prompt = """

Respond with only the Answer:
Name an 7-letter animal or other living creature that ends with the letter 'd'.
"""

for i in range(20):
    temperature = random.uniform(1, 1.5)
    print(temperature)
    output = get_output(prompt, temperature)
    print(output.choices[0].message.content)

