import collections 
import collections.abc
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def add_slide(prs, title, content_bullets, image_path=None):
    # Use layout 1 (Title and Content) or layout 5 (Title only) if image is big
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    
    title_shape = slide.shapes.title
    title_shape.text = title
    
    # Format Title
    title_shape.text_frame.paragraphs[0].font.size = Pt(36)
    title_shape.text_frame.paragraphs[0].font.bold = True
    title_shape.text_frame.paragraphs[0].font.color.rgb = RGBColor(0, 51, 102)

    body_shape = slide.shapes.placeholders[1]
    tf = body_shape.text_frame
    
    for i, bullet in enumerate(content_bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = bullet
        p.font.size = Pt(18)
        p.level = 0
        if ":" in bullet:
            # Simple hack to bold the part before the colon
            pass

    if image_path:
        # Resize text box to make room for image on the right
        body_shape.width = Inches(4.5)
        # Add image on the right
        try:
            slide.shapes.add_picture(image_path, Inches(5.0), Inches(2.0), width=Inches(4.5))
        except Exception as e:
            print(f"Could not load image {image_path}: {e}")

def create_presentation():
    prs = Presentation()
    
    # Title Slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Indian Language Identification using Audio Foundation Models"
    subtitle.text = "Exploring Zero-Shot, Prompt Learning, and Cross-Domain Generalization"
    
    # Slide 2: Zero-Shot CLAP
    add_slide(prs, "1. Zero-Shot CLAP (Baseline)", [
        "Architecture: Base CLAP (HTSAT Audio Encoder + RoBERTa Text Encoder).",
        "Modification from CLAP: None. This is the exact pre-trained architecture.",
        "How it works: Uses a static, hard-coded text prompt ('A person speaking in {language}').",
        "In-Domain Accuracy (Ekstep): ~75-80%",
        "Cross-Domain Accuracy (IITMandi): Severely degraded.",
        "Conclusion: Strong baseline but rigid prompt structure limits adaptation."
    ])

    # Slide 3: CoOp (Context Optimization)
    add_slide(prs, "2. CoOp (Context Optimization)", [
        "Architecture: Continuous Prompt Learning.",
        "Modification from CLAP: The discrete English words in the text prompt are replaced by learnable continuous vectors (V1, V2, ...).",
        "How it works: The Audio and Text encoders are 100% frozen. Only the continuous prompt vectors are trained via backpropagation on the Ekstep dataset.",
        "In-Domain Accuracy (Ekstep): 53.17%",
        "Cross-Domain Accuracy (IITMandi): 28.40%",
        "Conclusion: Static learned vectors easily overfit to the training domain's specific acoustic signature."
    ])

    # Slide 4: CoCoOp (Conditional Context Optimization)
    add_slide(prs, "3. CoCoOp (Conditional Context Optimization)", [
        "Architecture: Dynamic Audio-Conditioned Prompting.",
        "Modification from CLAP: Introduces a lightweight neural 'Meta-Net'.",
        "How it works: The Meta-Net takes the specific Audio Embedding as input and generates a dynamic bias vector, which is added to the CoOp text tokens. The prompt adapts to every single audio clip.",
        "In-Domain Accuracy (Ekstep): 89.58%",
        "Cross-Domain Accuracy (IITMandi): 24.75%",
        "Conclusion: Incredibly powerful for in-domain data, but suffers from Catastrophic Domain Overfitting when faced with background noise."
    ])
    
    # Slide 5: PALM (Prompt Alignment Learning)
    add_slide(prs, "4. PALM (Prompt Alignment Learning)", [
        "Architecture: Joint Space Alignment.",
        "Modification from CLAP: Adds an adapter layer to the text encoder to force alignment between the audio and text spaces.",
        "How it works: Uses few-shot prompt learning combined with a cross-modal alignment objective to bridge the modality gap.",
        "In-Domain Accuracy (Ekstep): 76.05%",
        "Cross-Domain Accuracy (IITMandi): 25.66%",
        "Conclusion: Better than base CoOp, but still succumbs to cross-domain acoustic shift."
    ])

    # Slide 6: Alternative Encoders (Whisper / WaveLM)
    add_slide(prs, "5. Supervised Acoustic Baselines", [
        "Architecture: Pure Acoustic Encoders (Whisper, WaveLM, Wav2Vec2, DistilHuBERT).",
        "Modification from CLAP: Completely discards the Text Encoder and contrastive learning. Replaces it with a Supervised Logistic Regression Probe on top of frozen audio features.",
        "Whisper (In-Domain -> Cross-Domain): 95.76% -> 82.77%",
        "WaveLM (In-Domain -> Cross-Domain): 95.32% -> 80.81%",
        "Wav2Vec2 (In-Domain -> Cross-Domain): 92.91% -> 52.10%",
        "Conclusion: Massive pre-trained acoustic models retain highly robust features across domain shifts, proving the vulnerability lies in CLAP's Audio-Text alignment mechanism."
    ])

    # Slide 7: XAI Interpretability (LIME)
    add_slide(prs, "6. XAI Interpretability: Text Prompt (LIME)", [
        "Objective: Why did the Text Prompts fail in cross-domain?",
        "Method: Applied LIME feature attribution to the Zero-Shot text prompt.",
        "Results (Weights): 'Marathi' (+0.0044), 'speaking' (+0.0007), 'person' (-0.0015).",
        "Analysis: The text encoder is working perfectly. It applies maximum positive weight to the actual class label (Marathi) and ignores grammatical fluff.",
        "Conclusion: The failure is NOT a text-understanding issue."
    ], image_path="/Users/shubhamdikshit/.gemini/antigravity/brain/1cfe90c1-cfc1-4897-8006-5066fafcb903/lime_plot.png")

    # Slide 8: XAI Interpretability (t-SNE)
    add_slide(prs, "7. XAI Interpretability: Acoustic Shift (t-SNE)", [
        "Objective: Visualizing the latent space alignment.",
        "Method: t-SNE plot of Ekstep Audio (Blue), YouTube Audio (Red), and Text Prompts (Black).",
        "Analysis: During training, CoOp/CoCoOp learned prompt vectors that anchor directly to the In-Domain (Ekstep) audio clusters.",
        "The Failure: When tested on YouTube data, the acoustic background noise physically shifted the audio embeddings to a completely different region of the latent space.",
        "Final Conclusion: The rigid text prompts were left stranded in the latent space, misaligned from the new noisy audio data."
    ], image_path="/Users/shubhamdikshit/.gemini/antigravity/brain/1cfe90c1-cfc1-4897-8006-5066fafcb903/tsne_domain_shift.png")

    prs.save("CLAP_IL_Presentation.pptx")
    print("Presentation saved as CLAP_IL_Presentation.pptx")

if __name__ == "__main__":
    create_presentation()
