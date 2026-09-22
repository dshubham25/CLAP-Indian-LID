import base64
import requests
import json

INIT_STR = "%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '48px', 'fontFamily': 'arial', 'primaryColor': '#e1f5fe', 'lineColor': '#0277bd'}}}%%\n"

m_str = INIT_STR + """graph LR
A(Music Audio) --> B[M-ResNet-50 / M-AST\nAudio Tower] --> C(Music\nEmbedding)
D(Natural Language\nDescriptions) --> E[BERT-based\nText Tower] --> F(Text\nEmbedding)
C --> G((Contrastive\nAlignment Loss))
F --> G
style G fill:#ffcdd2,stroke:#c62828,stroke-width:4px
"""

payload = {"code": m_str, "mermaid": {"theme": "default"}}
json_str = json.dumps(payload)
b64 = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8')
url = f"https://mermaid.ink/img/{b64}?bgColor=ffffff"

try:
    response = requests.get(url)
    response.raise_for_status()
    with open("mulan_arch.png", "wb") as f:
        f.write(response.content)
    print("Downloaded high-res mulan_arch.png")
except Exception as e:
    print(f"Failed to download: {e}")
