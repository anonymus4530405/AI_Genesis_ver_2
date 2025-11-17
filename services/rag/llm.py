import os
from groq import Groq

def groq_llm(prompt: str, temperature: float = 0.1):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY missing")
    
    client = Groq(api_key=key)
    model_name = "groq/compound"  # ensure this is valid

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {e}")
