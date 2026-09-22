import base64
import requests
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# Mermaid diagram definitions
diagrams = {
    "zero_shot": """graph LR
A[Audio Input] --> B[Frozen Audio Encoder] --> C[Audio Embedding]
D[Text: 'This is Hindi'] --> E[Frozen Text Encoder] --> F[Text Embedding]
C --> G((Cosine Similarity))
F --> G
G --> H[Prediction]
""",
    "coop": """graph LR
A[Audio Input] --> B[Frozen Audio Encoder] --> C[Audio Embedding]
D[Learnable V1..Vn + 'Hindi'] --> E[Frozen Text Encoder] --> F[Text Embedding]
C --> G((Cosine Similarity))
F --> G
G --> H[Prediction]
style D fill:#f9f,stroke:#333,stroke-width:2px
""",
    "cocoop": """graph LR
A[Audio Input] --> B[Frozen Audio Encoder] --> C[Audio Embedding]
C --> D[Meta-Net] --> E[Dynamic Bias]
F[Learnable Vectors V1..Vn] --> G((Add))
E --> G
G --> H[Conditioned Prompt] --> I[Frozen Text Encoder] --> J[Text Embedding]
C --> K((Cosine Similarity))
J --> K
style D fill:#f9f,stroke:#333,stroke-width:2px
""",
    "palm": """graph LR
A[Audio Input] --> B[Frozen Audio Encoder] --> C[Audio Embedding]
D[Prompt] --> E[Text Encoder] --> F[Alignment Adapter] --> G[Text Embedding]
C --> H((Cosine Similarity))
G --> H
style F fill:#f9f,stroke:#333,stroke-width:2px
""",
    "alt_encoders": """graph LR
A[Audio Input] --> B[Supervised Audio Encoder e.g. Whisper] --> C[Acoustic Features]
C --> D[Supervised Linear Probe] --> E[Prediction]
style D fill:#f9f,stroke:#333,stroke-width:2px
"""
}

# Download images via mermaid.ink
for name, m_str in diagrams.items():
    # mermaid.ink requires the payload to be a base64 encoded JSON string
    payload = {"code": m_str, "mermaid": {"theme": "default"}}
    json_str = json.dumps(payload)
    b64 = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8')
    url = f"https://mermaid.ink/img/{b64}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(f"{name}_arch.png", "wb") as f:
            f.write(response.content)
        print(f"Downloaded {name}_arch.png")
    except Exception as e:
        print(f"Failed to download {name}: {e}")

# Build Presentation
prs = Presentation()

# Title Slide
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Indian Language Identification using Audio Foundation Models"
slide.placeholders[1].text = "Visual Architecture Exploration"

def add_visual_slide(title, image_file, bullet_points):
    slide = prs.slides.add_slide(prs.slide_layouts[5]) # Title only layout
    slide.shapes.title.text = title
    
    # Add Architecture Image in the center
    try:
        # Positioning: Left 0.5, Top 1.5, Width 9
        slide.shapes.add_picture(image_file, Inches(0.5), Inches(1.5), width=Inches(9.0))
    except Exception as e:
        print(f"Could not load {image_file}: {e}")
        
    # Add text box below for results
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(5.0), Inches(9.0), Inches(2.0))
    tf = txBox.text_frame
    tf.word_wrap = True
    
    for i, point in enumerate(bullet_points):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = point
        p.font.size = Pt(16)

add_visual_slide(
    "1. Zero-Shot CLAP (Baseline)",
    "zero_shot_arch.png",
    ["Results: Strong In-Domain (~75%), but fails completely in Cross-Domain due to rigid text prompts."]
)

add_visual_slide(
    "2. CoOp (Context Optimization)",
    "coop_arch.png",
    ["Modification: Replaces discrete words with continuous learnable vectors (Pink).",
     "Results: 53.17% (In-Domain) | 28.40% (Cross-Domain). Succumbs to Catastrophic Domain Overfitting."]
)

add_visual_slide(
    "3. CoCoOp (Conditional Context Optimization)",
    "cocoop_arch.png",
    ["Modification: Introduces Meta-Net (Pink) to dynamically adapt the prompt to each audio instance.",
     "Results: Near SOTA 89.58% (In-Domain) | 24.75% (Cross-Domain). Highly powerful but heavily reliant on domain-specific acoustic features."]
)

add_visual_slide(
    "4. PALM (Prompt Alignment Learning)",
    "palm_arch.png",
    ["Modification: Adds a learnable adapter (Pink) to force alignment between Audio and Text spaces.",
     "Results: 76.05% (In-Domain) | 25.66% (Cross-Domain)."]
)

add_visual_slide(
    "5. Alternative Acoustic Encoders",
    "alt_encoders_arch.png",
    ["Modification: Abandons the Contrastive Text Encoder entirely for a pure Supervised Linear Probe (Pink).",
     "Results: Whisper dropped only slightly (95% -> 82%) in cross-domain, proving the failure lies in the rigid Prompt-Audio alignment, not just acoustic degradation."]
)

# Keep the XAI slides from before (with images on the right)
def add_xai_slide(title, image_path, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    body = slide.shapes.placeholders[1]
    tf = body.text_frame
    body.width = Inches(4.5)
    for i, b in enumerate(bullets):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = b
        p.font.size = Pt(16)
    try:
        slide.shapes.add_picture(image_path, Inches(5.0), Inches(2.0), width=Inches(4.5))
    except Exception as e:
        pass

add_xai_slide(
    "6. XAI: Why did prompts fail? (LIME)",
    "/Users/shubhamdikshit/.gemini/antigravity/brain/1cfe90c1-cfc1-4897-8006-5066fafcb903/lime_plot.png",
    ["Text Encoder analysis: Did the text prompt fail?", 
     "LIME proves it assigned max positive weight to 'Marathi'.",
     "Conclusion: The text prompt functioned perfectly."]
)

add_xai_slide(
    "7. XAI: Acoustic Shift (t-SNE)",
    "/Users/shubhamdikshit/.gemini/antigravity/brain/1cfe90c1-cfc1-4897-8006-5066fafcb903/tsne_domain_shift.png",
    ["Visualizing the Latent Space.",
     "The YouTube Audio (Red) physically shifted away from the prompt anchors due to background noise.",
     "Final Conclusion: Static text prompts misaligned with shifting audio embeddings."]
)

prs.save("CLAP_IL_Visual_Presentation.pptx")
print("Visual Presentation saved as CLAP_IL_Visual_Presentation.pptx")
