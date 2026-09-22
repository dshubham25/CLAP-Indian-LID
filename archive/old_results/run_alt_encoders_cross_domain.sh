#!/bin/bash
echo "Starting Sequential Cross-Domain Evaluation for Alternative Encoders on IITMandi_YouTube..."

echo "1/3: Evaluating Whisper..."
python evaluate_alt_encoders.py --model whisper --test-dataset IITMandi_YouTube | tee whisper_cross_domain.log

echo "2/3: Evaluating WaveLM..."
python evaluate_alt_encoders.py --model wavlm --test-dataset IITMandi_YouTube | tee wavlm_cross_domain.log

echo "3/3: Evaluating Wav2Vec2..."
python evaluate_alt_encoders.py --model wav2vec2 --test-dataset IITMandi_YouTube | tee wav2vec2_cross_domain.log

echo "All alternative encoder evaluations finished!"
