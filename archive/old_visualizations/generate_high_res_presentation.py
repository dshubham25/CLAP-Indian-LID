import base64
import requests
import json
from pptx import Presentation
from pptx.util import Inches, Pt

# Massive font size so when the image is scaled into the slide, the text remains extremely readable
INIT_STR = "%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '48px', 'fontFamily': 'arial', 'primaryColor': '#e1f5fe', 'lineColor': '#0277bd'}}}%%\n"

diagrams = {
    "zero_shot": INIT_STR + """graph LR
A(Audio) --> B[Frozen HTSAT\nAudio Encoder] --> C(Audio\nEmbedding)
D(Text:\n'This is Hindi') --> E[Frozen RoBERTa\nText Encoder] --> F(Text\nEmbedding)
C --> G((Cosine\nSimilarity))
F --> G
G --> H{Prediction}
""",
    "coop": INIT_STR + """graph LR
A(Audio) --> B[Frozen HTSAT\nAudio Encoder] --> C(Audio\nEmbedding)
D(Learnable Vectors\nV1..Vn + 'Hindi') --> E[Frozen RoBERTa\nText Encoder] --> F(Text\nEmbedding)
C --> G((Cosine\nSimilarity))
F --> G
G --> H{Prediction}
style D fill:#ffcdd2,stroke:#c62828,stroke-width:4px
""",
    "cocoop": INIT_STR + """graph LR
A(Audio) --> B[Frozen HTSAT\nAudio Encoder] --> C(Audio\nEmbedding)
C --> D[Meta-Net\nNeural Network] --> E(Dynamic\nBias)
F(Learnable Vectors\nV1..Vn) --> G((Add))
E --> G
G --> H(Conditioned\nPrompt) --> I[Frozen RoBERTa\nText Encoder] --> J(Text\nEmbedding)
C --> K((Cosine\nSimilarity))
J --> K
style D fill:#ffcdd2,stroke:#c62828,stroke-width:4px
""",
    "palm": INIT_STR + """graph LR
A(Audio) --> B[Frozen HTSAT\nAudio Encoder] --> C(Audio\nEmbedding)
D(Prompt) --> E[RoBERTa\nText Encoder] --> F[Learnable\nAlignment Adapter] --> G(Text\nEmbedding)
C --> H((Cosine\nSimilarity))
G --> H
style F fill:#ffcdd2,stroke:#c62828,stroke-width:4px
""",
    "alt_encoders": INIT_STR + """graph LR
A(Audio) --> B[Massive Audio Encoder\nWhisper / WaveLM] --> C(Acoustic\nFeatures)
C --> D[Supervised\nLinear Probe] --> E{Prediction}
style D fill:#ffcdd2,stroke:#c62828,stroke-width:4px
"""
}

# Download images via mermaid.ink
for name, m_str in diagrams.items():
    payload = {"code": m_str, "mermaid": {"theme": "default"}}
    json_str = json.dumps(payload)
    b64 = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8')
    url = f"https://mermaid.ink/img/{b64}?bgColor=ffffff" # add white background
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(f"{name}_arch.png", "wb") as f:
            f.write(response.content)
        print(f"Downloaded high-res {name}_arch.png")
    except Exception as e:
        print(f"Failed to download {name}: {e}")

# Build Presentation
prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Indian Language Identification using Audio Foundation Models"
slide.placeholders[1].text = "Visual Architecture Exploration"

def add_visual_slide(title, image_file, bullet_points):
    slide = prs.slides.add_slide(prs.slide_layouts[5]) 
    slide.shapes.title.text = title
    
    try:
        # Scale image to maximum width possible (9.5 inches out of 10)
        slide.shapes.add_picture(image_file, Inches(0.25), Inches(1.5), width=Inches(9.5))
    except Exception as e:
        print(f"Could not load {image_file}: {e}")
        
    # Push text box lower (to 5.5 inches) to give image more room
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(5.5), Inches(9.0), Inches(1.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    
    for i, point in enumerate(bullet_points):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = point
        p.font.size = Pt(18)

add_visual_slide("1. Zero-Shot CLAP (Baseline)", "zero_shot_arch.png", ["Results: Strong In-Domain (~75%), but fails completely in Cross-Domain due to rigid text prompts."])
add_visual_slide("2. CoOp (Context Optimization)", "coop_arch.png", ["Modification: Replaces discrete words with continuous learnable vectors (Pink).", "Results: 53.17% (In-Domain) | 28.40% (Cross-Domain). Succumbs to Catastrophic Domain Overfitting."])
add_visual_slide("3. CoCoOp (Conditional Context Optimization)", "cocoop_arch.png", ["Modification: Introduces Meta-Net (Pink) to dynamically adapt the prompt to each audio instance.", "Results: Near SOTA 89.58% (In-Domain) | 24.75% (Cross-Domain). Highly powerful but heavily reliant on domain-specific acoustic features."])
add_visual_slide("4. PALM (Prompt Alignment Learning)", "palm_arch.png", ["Modification: Adds a learnable adapter (Pink) to force alignment between Audio and Text spaces.", "Results: 76.05% (In-Domain) | 25.66% (Cross-Domain)."])
add_visual_slide("5. Alternative Acoustic Encoders", "alt_encoders_arch.png", ["Modification: Abandons the Contrastive Text Encoder entirely for a pure Supervised Linear Probe (Pink).", "Results: Whisper dropped only slightly (95% -> 82%) in cross-domain, proving the failure lies in the rigid Prompt-Audio alignment, not just acoustic degradation."])

def add_xai_slide(title, image_path, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    body = slide.shapes.placeholders[1]
    tf = body.text_frame
    body.width = Inches(4.5)
    for i, b in enumerate(bullets):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = b
        p.font.size = Pt(18)
    try:
        slide.shapes.add_picture(image_path, Inches(5.0), Inches(2.0), width=Inches(4.5))
    except Exception as e:
        pass

add_xai_slide("6. XAI: Why did prompts fail? (LIME)", "/Users/shubhamdikshit/.gemini/antigravity/brain/1cfe90c1-cfc1-4897-8006-5066fafcb903/lime_plot.png", ["Text Encoder analysis: Did the text prompt fail?", "LIME proves it assigned max positive weight to 'Marathi'.", "Conclusion: The text prompt functioned perfectly."])
add_xai_slide("7. XAI: Acoustic Shift (t-SNE)", "/Users/shubhamdikshit/.gemini/antigravity/brain/1cfe90c1-cfc1-4897-8006-5066fafcb903/tsne_domain_shift.png", ["Visualizing the Latent Space.", "The YouTube Audio (Red) physically shifted away from the prompt anchors due to background noise.", "Final Conclusion: Static text prompts misaligned with shifting audio embeddings."])

prs.save("CLAP_IL_Visual_Presentation_HD.pptx")
print("Saved as CLAP_IL_Visual_Presentation_HD.pptx")
