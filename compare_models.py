"""
Multi-Model Evaluation & Benchmarking for Small / Resource-Constrained Systems.
Compares Qwen 2.5 (0.5B, 1.5B, 3B) and Llama 3.2 (1B) for Speed, Accuracy, and JSON Compliance.
"""

import time
import json
import openpyxl
from typing import List, Dict, Any
from benchmark import load_dataset
from fast_extractor import extract_fast_summary, format_prompt_input
from classifier import FinancialClassifier

MODELS_TO_TEST = [
    "qwen2.5:0.5b",
    "llama3.2:1b",
    "qwen2.5:1.5b",
    "qwen2.5:3b"
]

def run_model_comparison(samples_count: int = 12, use_offline: bool = True):
    dataset = load_dataset()
    
    # Balanced selection
    b_samples = [r for r in dataset if "B" in str(r.get("Business model", ""))][:samples_count // 2]
    other_samples = [r for r in dataset if "B" not in str(r.get("Business model", ""))][:samples_count - len(b_samples)]
    test_batch = b_samples + other_samples
    
    print(f"Loaded {len(test_batch)} test samples ({len(b_samples)} Model B, {len(other_samples)} Non-B).")
    
    # Pre-build inputs
    inputs = []
    for item in test_batch:
        name = item.get("Company Name", "Unknown")
        url = item.get("Website", "")
        gt_model = item.get("Business model", "").strip()
        gt_is_b = "B" in gt_model
        
        if use_offline:
            text = f"Company: {name}\nWebsite: {url}\nDescription: {item.get('What they do', '')}\nEvidence: {item.get('Evidence', '')}"
        else:
            summary = extract_fast_summary(url)
            text = format_prompt_input(summary, company_name=name)
            
        inputs.append({
            "name": name,
            "url": url,
            "ground_truth": gt_model,
            "is_b": gt_is_b,
            "prompt_text": text
        })
        
    summary_results = []
    
    for model_name in MODELS_TO_TEST:
        print("\n" + "="*80)
        print(f"Evaluating Model: {model_name} on {len(inputs)} companies...")
        print("="*80)
        
        classifier = FinancialClassifier(provider="ollama", model=model_name)
        
        tp_b, fp_b, fn_b, tn_b = 0, 0, 0, 0
        json_valid = 0
        total_time = 0.0
        
        for idx, sample in enumerate(inputs, 1):
            t0 = time.time()
            res = classifier.classify_text(sample["prompt_text"])
            dt = time.time() - t0
            total_time += dt
            
            if res.get("success"):
                json_valid += 1
                pred = res["data"]
                pred_model = pred.get("business_model", "")
                pred_is_b = "B" in pred_model
                
                if sample["is_b"] and pred_is_b:
                    tp_b += 1
                elif not sample["is_b"] and pred_is_b:
                    fp_b += 1
                elif sample["is_b"] and not pred_is_b:
                    fn_b += 1
                else:
                    tn_b += 1
                    
                match_str = "MATCH" if (sample["is_b"] == pred_is_b) else "MISMATCH"
                print(f"  [{idx:02d}] {sample['name'][:25]:<25} | GT: {sample['ground_truth'][:15]:<15} | Pred: {pred_model[:15]:<15} | {match_str} ({dt:.2f}s)")
            else:
                if sample["is_b"]:
                    fn_b += 1
                else:
                    tn_b += 1
                print(f"  [{idx:02d}] {sample['name'][:25]:<25} | JSON ERROR: {res.get('error')} ({dt:.2f}s)")
                
        avg_latency = total_time / len(inputs) if inputs else 0
        prec_b = (tp_b / (tp_b + fp_b)) if (tp_b + fp_b) > 0 else 0
        rec_b = (tp_b / (tp_b + fn_b)) if (tp_b + fn_b) > 0 else 0
        f1_b = (2 * prec_b * rec_b / (prec_b + rec_b)) if (prec_b + rec_b) > 0 else 0
        json_rate = (json_valid / len(inputs)) * 100
        
        summary_results.append({
            "model": model_name,
            "avg_latency_s": round(avg_latency, 2),
            "precision_b": round(prec_b * 100, 1),
            "recall_b": round(rec_b * 100, 1),
            "f1_b": round(f1_b * 100, 1),
            "json_compliance": round(json_rate, 1)
        })
        
    print("\n" + "="*80)
    print("FINAL COMPARISON ON SMALL / CPU SYSTEMS")
    print("="*80)
    print(f"{'Model Name':<16} | {'Avg Latency':<12} | {'Recall (B)':<11} | {'Precision (B)':<13} | {'F1-Score':<10} | {'JSON Schema':<11}")
    print("-" * 85)
    for r in summary_results:
        print(f"{r['model']:<16} | {r['avg_latency_s']:>7.2f}s     | {r['recall_b']:>7.1f}%    | {r['precision_b']:>9.1f}%     | {r['f1_b']:>6.1f}%    | {r['json_compliance']:>8.1f}%")
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    run_model_comparison(samples_count=args.samples, use_offline=not args.live)
