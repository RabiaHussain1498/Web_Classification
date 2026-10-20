"""
Benchmarking Harness for Financial Business Model Classification.
Evaluates accuracy, latency, and Model B detection against Finance - Sample List.xlsx.
"""

import os
import json
import time
import argparse
import openpyxl
from typing import List, Dict, Any
from fast_extractor import extract_fast_summary, format_prompt_input
from classifier import FinancialClassifier

CACHE_FILE = "scrape_cache.json"

def load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache: Dict[str, Any]):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def load_dataset(file_path: str = "Finance - Sample List.xlsx") -> List[Dict[str, Any]]:
    """Load ground truth rows from sample list."""
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active
    headers = [c.value for c in ws[1]]
    
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r[0]:
            continue
        item = {headers[i]: r[i] for i in range(len(headers)) if i < len(r)}
        rows.append(item)
    return rows

def run_benchmark(
    samples_count: int = 10,
    provider: str = "ollama",
    model: str = "qwen2.5:7b",
    endpoint: str = "http://localhost:11434",
    api_key: str = "",
    use_offline_text: bool = False
):
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} total samples from dataset.")
    
    # Stratified selection to ensure Model B samples are included
    b_samples = [r for r in dataset if "B" in str(r.get("Business model", ""))][:samples_count // 2]
    other_samples = [r for r in dataset if "B" not in str(r.get("Business model", ""))][:samples_count - len(b_samples)]
    test_batch = b_samples + other_samples
    
    print(f"Selected {len(test_batch)} test samples ({len(b_samples)} Model B, {len(other_samples)} other models).")
    
    classifier = FinancialClassifier(
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_key=api_key
    )
    
    cache = load_cache()
    results = []
    
    tp_b, fp_b, fn_b, tn_b = 0, 0, 0, 0
    total_time = 0.0
    
    print("\n" + "="*80)
    print(f"Starting Benchmark on {len(test_batch)} companies...")
    print("="*80)
    
    for idx, item in enumerate(test_batch, 1):
        name = item.get("Company Name", "Unknown")
        url = item.get("Website", "")
        ground_truth_model = item.get("Business model", "").strip()
        ground_truth_is_b = "B" in ground_truth_model
        
        print(f"\n[{idx}/{len(test_batch)}] {name} ({url})")
        print(f"  Ground Truth: {ground_truth_model}")
        
        if use_offline_text:
            # Test purely prompt logic using existing description in dataset
            input_text = f"Company: {name}\nWebsite: {url}\nDescription: {item.get('What they do', '')}\nEvidence: {item.get('Evidence', '')}"
        else:
            # Real web extraction with caching
            if url in cache:
                summary = cache[url]
            else:
                summary = extract_fast_summary(url)
                cache[url] = summary
                save_cache(cache)
                
            input_text = format_prompt_input(summary, company_name=name)
            
        t0 = time.time()
        res = classifier.classify_text(input_text)
        t_elapsed = time.time() - t0
        total_time += t_elapsed
        
        if res.get("success"):
            pred = res["data"]
            pred_model = pred.get("business_model", "")
            pred_is_b = "B" in pred_model
            match = (ground_truth_is_b == pred_is_b)
            
            if ground_truth_is_b and pred_is_b:
                tp_b += 1
            elif not ground_truth_is_b and pred_is_b:
                fp_b += 1
            elif ground_truth_is_b and not pred_is_b:
                fn_b += 1
            else:
                tn_b += 1
                
            print(f"  Prediction  : {pred_model} (Build: {pred.get('build')}) [Time: {t_elapsed:.2f}s]")
            print(f"  Match Model B: {'CORRECT' if match else 'MISMATCH'}")
            print(f"  Evidence    : {pred.get('build_evidence', '')}")
            
            results.append({
                "company": name,
                "url": url,
                "ground_truth": ground_truth_model,
                "predicted": pred_model,
                "predicted_build": pred.get("build"),
                "time_sec": round(t_elapsed, 2),
                "matched_b": match
            })
        else:
            print(f"  Error: {res.get('error')} [Time: {t_elapsed:.2f}s]")
            results.append({
                "company": name,
                "url": url,
                "ground_truth": ground_truth_model,
                "predicted": "ERROR",
                "error": res.get("error"),
                "time_sec": round(t_elapsed, 2),
                "matched_b": False
            })

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    avg_latency = total_time / len(test_batch) if test_batch else 0
    precision_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
    recall_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
    f1_b = (2 * precision_b * recall_b) / (precision_b + recall_b) if (precision_b + recall_b) > 0 else 0
    
    print(f"Total Evaluated     : {len(test_batch)}")
    print(f"Avg Latency Per Site: {avg_latency:.2f} seconds")
    print(f"Model B True Pos    : {tp_b}")
    print(f"Model B False Pos   : {fp_b}")
    print(f"Model B False Neg   : {fn_b}")
    print(f"Model B True Neg    : {tn_b}")
    print(f"Model B Precision   : {precision_b*100:.1f}%")
    print(f"Model B Recall      : {recall_b*100:.1f}%")
    print(f"Model B F1 Score    : {f1_b*100:.1f}%")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Classifier Benchmark")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "openai", "mock"])
    parser.add_argument("--model", type=str, default="qwen2.5:7b")
    parser.add_argument("--endpoint", type=str, default="http://localhost:11434")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--offline", action="store_true", help="Use offline text descriptions instead of live scraping")
    
    args = parser.parse_args()
    run_benchmark(
        samples_count=args.samples,
        provider=args.provider,
        model=args.model,
        endpoint=args.endpoint,
        api_key=args.api_key,
        use_offline_text=args.offline
    )
