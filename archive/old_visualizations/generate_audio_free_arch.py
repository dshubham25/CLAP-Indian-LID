import base64
import requests
import json

INIT_STR = "%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '48px', 'fontFamily': 'arial', 'primaryColor': '#e1f5fe', 'lineColor': '#0277bd'}}}%%\n"

m_str = INIT_STR + """graph LR
A(Class Names) --> B[Multi-grained\nPrompt Constructor]
B --> C(Continuous\nPrompts)
C --> D[Frozen\nText Encoder] --> E(Text\nEmbedding)
E --> F((Text-only\nRegularization))
style B fill:#ffcdd2,stroke:#c62828,stroke-width:4px
style F fill:#ffcdd2,stroke:#c62828,stroke-width:4px
"""

payload = {"code": m_str, "mermaid": {"theme": "default"}}
json_str = json.dumps(payload)
b64 = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8')
url = f"https://mermaid.ink/img/{b64}?bgColor=ffffff"

try:
    response = requests.get(url)
    response.raise_for_status()
    with open("audio_free_arch.png", "wb") as f:
        f.write(response.content)
    print("Downloaded high-res audio_free_arch.png")
except Exception as e:
    print(f"Failed to download: {e}")
