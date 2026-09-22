#!/bin/bash
echo "Starting Sequential Cross-Domain Evaluation on IITMandi_YouTube..."

echo "1/4: Evaluating DistilHuBERT..."
python evaluate_alt_encoders.py --model distilhubert --test-dataset IITMandi_YouTube | tee distilhubert_cross_domain.log

echo "2/4: Evaluating CoOp..."
python evaluate_cross_domain.py --model coop | tee coop_cross_domain.log

echo "3/4: Evaluating PALM..."
python evaluate_cross_domain.py --model palm | tee palm_cross_domain.log

echo "4/4: Evaluating CoCoOp..."
python evaluate_cross_domain.py --model cocoop | tee cocoop_cross_domain.log

echo "All evaluations finished!"
