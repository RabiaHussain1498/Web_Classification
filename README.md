# Financial Business Model Classifier & Benchmark Pipeline

This system implements the **ArhamSoft Finance Account Classifier (v1.0)** for processing 60,000+ financial company URLs. It provides ultra-fast shallow web extraction, strict JSON classification into Models A–E, and an evaluation benchmark to identify **Model B (Product-Led)** institutions with high precision and low latency.

---

## 1. System Components

* **`prompt_config.py`**:
  * Formatted decision tree for Models **A through E** and controlled taxonomy (27 institution types, 9 sector groups, 4 product types).
  * System prompt engineered to output compact JSON (< 120 tokens response) for maximum inference speed on local LLMs.
* **`fast_extractor.py`**:
  * Shallow scraper fetching only high-signal markers (`<title>`, `<meta description>`, headings, app store links, developer API docs).
  * Limits context to ~250–350 tokens, eliminating the need to crawl 5–7 deep pages upfront.
  * Resilient DNS, subdomain fallback, and SSL error handling.
* **`classifier.py`**:
  * Connects to **Ollama** (`/api/chat` with JSON format enforcement) or any **OpenAI-compatible** endpoint (vLLM, LiteLLM, OpenRouter).
  * Robust fallback JSON parsing and mock mode for testing without GPU.
* **`benchmark.py`**:
  * Evaluates accuracy, Precision, Recall, F1-Score, and latency per site against the 860 rows in `Finance - Sample List.xlsx`.
  * Built-in file caching (`scrape_cache.json`) to prevent repeated network requests during prompt iteration.

---

## 2. Quickstart & Usage

### A. Testing Locally with Mock Mode (Dry Run)
Verify the benchmark pipeline logic without a running LLM:
```bash
python benchmark.py --samples 10 --offline --provider mock
```

### B. Running with Ollama (e.g. on Ubuntu / Linux with GPU)
1. Ensure Ollama is running:
   ```bash
   ollama run qwen2.5:7b
   ```
2. Run benchmark on sample list:
   ```bash
   python benchmark.py --samples 20 --provider ollama --model qwen2.5:7b --endpoint http://localhost:11434
   ```

### C. Running via OpenAI / vLLM / LiteLLM API
```bash
python benchmark.py --samples 20 --provider openai --model gpt-4o-mini --endpoint https://api.openai.com/v1 --api_key YOUR_KEY
```

---

## 3. Business Model Decision Tree (Taxonomy)

| Business Model | What it Sells | Build Layer | Description |
| :--- | :--- | :--- | :--- |
| **A - Services-led** | Financial product | `Vendor` / `Resells` | Sells financial products on vendor, partner, or white-label technology (e.g., community banks using Fiserv/Jack Henry cores, independent insurance agents). |
| **B - Product-led** | Financial product | `Own` | Sells financial products through an **in-house, proprietary customer-facing app, platform, exchange, or API** (e.g., Neobanks, proprietary digital lenders, crypto exchanges, custom payment networks). |
| **C - Software vendor** | Software | `Own` | Sells financial software, developer APIs, or data feeds to businesses; technology *is* the product. |
| **D - Capital & advisory-led** | Capital & advisory | `None` | Earns through investment capital management, asset ownership, wealth advisory, M&A advisory, or ratings. |
| **E - Tech & professional services** | Services | `None` | Provides outsourced technology development, IT services, consulting, or collections to other businesses. |
