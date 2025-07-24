from google import genai
import pandas as pd
import re
import json
import tiktoken # pip install tiktoken
import streamlit as st
import os

#Set Environment Variables

os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]


#Load Dataset:

path = "data.csv" 
df = pd.read_csv(path)
reviews = df["reviews.text"].iloc[:200].tolist()



#Tokenizer and Pricing:

enc = tiktoken.encoding_for_model("gpt-4")

pricing = {
"gpt-4.1-nano": {"input": 0.0004, "output": 0.0012},
"gpt-4.1-mini": {"input": 0.0010, "output": 0.0030}
}

def estimate_tokens(text):
    return len(enc.encode(text))

def estimate_cost(model, input_tokens, output_tokens):
    p = pricing[model]
    return (input_tokens / 1000 * p["input"]) + (output_tokens / 1000 * p["output"])



# Sentiment Analysis:

sentiment_prompt = f"Return a json list of 1s and 0s for each item on the list depending on whether a review demonstrates a positive or a negative sentiment {reviews}"
sent_input_tokens = estimate_tokens(sentiment_prompt)

client = genai.Client()

response = client.models.generate_content(model="gemini-2.5-flash", contents=sentiment_prompt)
clean_json = re.sub(r"^```json\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE)

data = json.loads(clean_json)

sent_output_text = response.text.strip()
sent_output_tokens = estimate_tokens(sent_output_text)

avg = sum(int(num) for num in data) / len(data)
print("Sentiment Score:", avg)

# Summary Generation:

summary_prompt = f"Generate summary from reviews {reviews}"
summary_input_tokens = estimate_tokens(summary_prompt)

summary_response = client.models.generate_content(model="gemini-2.5-flash", contents=summary_prompt)

summary_output_text = summary_response.text.strip()
summary_output_tokens = estimate_tokens(summary_output_text)



# Cost Estimation:

total_input_tokens = sent_input_tokens + summary_input_tokens
total_output_tokens = sent_output_tokens + summary_output_tokens

cost_nano = estimate_cost("gpt-4.1-nano", total_input_tokens, total_output_tokens)
cost_mini = estimate_cost("gpt-4.1-mini", total_input_tokens, total_output_tokens)

print("Token & Cost Summary:")
print("Input Tokens:", total_input_tokens)
print("Output Tokens:", total_output_tokens)
print("GPT-4.1 Nano Cost: $", round(cost_nano, 4))
print("GPT-4.1 Mini Cost: $", round(cost_mini, 4))



# Summary Report:

print(summary_output_text)