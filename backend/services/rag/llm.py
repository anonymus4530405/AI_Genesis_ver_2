import os
from groq import Groq

def groq_llm(prompt: str):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY missing")
    client = Groq(api_key=key)
    
    # <-- Use a currently supported model
    model_name = "groq/compound"  

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content
