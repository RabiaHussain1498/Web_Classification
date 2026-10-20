"""
Prompt Configuration & Decision Taxonomy for Financial Business Model Classification.
Based on ArhamSoft Finance Account Classifier v1.0 specification.
"""

FINANCE_TAXONOMY = {
    "business_models": [
        "A - Services-led",
        "B - Product-led",
        "C - Software vendor",
        "D - Capital & advisory-led",
        "E - Tech & professional services"
    ],
    "what_it_sells": [
        "Financial product",
        "Software",
        "Capital & advisory",
        "Services"
    ],
    "sector_groups": [
        "Banking",
        "Lending",
        "Payments & infrastructure",
        "Investing & wealth",
        "Insurance",
        "Capital & advisory",
        "Services firms",
        "Software vendors",
        "Marketplaces"
    ],
    "institution_types": [
        "Traditional Bank",
        "Digital Bank",
        "Neobank / Consumer Finance App",
        "Credit Union",
        "Custody & Trust Bank",
        "Traditional Lender / Servicer",
        "Online Lender / BNPL",
        "Payments & Card Processing",
        "Wealth Management / Brokerage",
        "Asset / Investment Manager",
        "Advisory / Investment Bank",
        "PE / VC / Private Credit",
        "Holding Company / Family Office / Asset Owner",
        "Insurance Company",
        "Insurance Broker / Agency / MGA",
        "Crypto & Digital Assets",
        "Exchange & Market Infrastructure",
        "Trading / Investing Platform",
        "Trading & Capital Markets Tech",
        "Embedded Finance / BaaS & Fintech Infrastructure",
        "Financial Data & Credit Bureaus",
        "Financial Marketplace / Comparison",
        "Fund Administration & Transfer Agency",
        "Accounting, Tax & Financial Advisory Firms",
        "Financial BPO & Collections",
        "Other Industry Software"
    ]
}

# Ultra-optimized, token-efficient system prompt for local LLMs (Qwen 2.5 / 3 8B, Llama 3, etc.)
SYSTEM_PROMPT = """You are an expert financial business intelligence classifier.
Your job is to analyze website context and classify the institution into one of 5 distinct business models and assign taxonomy attributes.

### DECISION TREE:
1. WHAT IT SELLS:
   - "Financial product": Accounts, cards, loans, mortgages, payments, deposits, insurance policies, trading access.
   - "Software": Financial software platforms, developer APIs, data feeds, SaaS solutions.
   - "Capital & advisory": Wealth management, asset management, private equity, investment banking, advisory, rating, research.
   - "Services": Outsourced IT, professional services, collections, consulting.

2. BUILD CLASSIFICATION (Customer-facing technology layer):
   - "Own": The company builds, owns, and operates its proprietary customer-facing platform, app, trading engine, or exchange.
   - "Vendor": The company relies on a third-party vendor core/white-label platform (e.g. Jack Henry, Fiserv, Q2, broker clearing firm) without proprietary consumer app.
   - "Resells": The company brokers or resells financial products from other institutions.
   - "None": Not applicable (typically Capital & advisory or standard services).

3. BUSINESS MODEL ASSIGNMENT:
   - "B - Product-led": Sells a financial product through an IN-HOUSE, PROPRIETARY customer-facing app, platform, exchange, or API.
     * EXAMPLES: FinTech neobanks (Chime, Revolut), crypto exchanges/protocols (Coinbase, DeFi), proprietary algorithmic/online lenders (Upstart), or custom payment networks.
     * EXCLUSIONS: Traditional community banks, regional banks, credit unions, mortgage brokers, and wealth advisors with standard web portals are NOT Model B.
   - "A - Services-led": Sells financial products on VENDOR, PARTNER, or WHITE-LABEL technology.
     * EXAMPLES: Traditional community banks, credit unions, mortgage brokers, insurance agencies, and financial advisor offices (e.g. Northwestern Mutual *.nm.com practices) using parent or vendor technology.
   - "C - Software vendor": Sells software, tech infrastructure, or data to businesses; the technology IS the product.
   - "D - Capital & advisory-led": Earns by managing/investing capital, holding assets, wealth advisory, PE/VC, asset management, investment banking.
   - "E - Tech & professional services": Sells technology services, consulting, or BPO to other businesses.

### RESPONSE FORMAT:
Respond ONLY with a valid JSON object matching this exact schema:
{
  "business_model": "A - Services-led" | "B - Product-led" | "C - Software vendor" | "D - Capital & advisory-led" | "E - Tech & professional services",
  "what_it_sells": "Financial product" | "Software" | "Capital & advisory" | "Services",
  "build": "Own" | "Vendor" | "Resells" | "None",
  "institution_type": "<Exact institution type from taxonomy>",
  "sector_group": "<Exact sector group from taxonomy>",
  "confidence": "High" | "Medium" | "Low",
  "build_evidence": "<Concise 1-sentence reason for build and business model>",
  "what_they_do": "<1-sentence summary of company operations>"
}
"""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Website: https://branches.bitcoindepot.com\nTitle: Bitcoin Depot - Bitcoin ATM Network\nContent: Buy Bitcoin and crypto instantly with cash at over 7,000 Bitcoin Depot kiosks and via our BDCheckout mobile app.",
        "output": {
            "business_model": "B - Product-led",
            "what_it_sells": "Financial product",
            "build": "Own",
            "institution_type": "Crypto & Digital Assets",
            "sector_group": "Payments & infrastructure",
            "confidence": "High",
            "build_evidence": "Operates proprietary crypto ATM network software and consumer mobile app.",
            "what_they_do": "Bitcoin ATM kiosk network and digital wallet app for purchasing cryptocurrency."
        }
    },
    {
        "input": "Website: https://1stiowa.bank\nTitle: First Iowa State Bank\nContent: Member FDIC. Equal Housing Lender. Offering personal checking, savings, commercial loans, and online banking powered by standard core vendor.",
        "output": {
            "business_model": "A - Services-led",
            "what_it_sells": "Financial product",
            "build": "Vendor",
            "institution_type": "Traditional Bank",
            "sector_group": "Banking",
            "confidence": "High",
            "build_evidence": "Traditional community bank utilizing third-party vendor online banking platform.",
            "what_they_do": "Community bank providing personal and commercial deposit accounts and lending."
        }
    },
    {
        "input": "Website: https://advisor.firstcommand.com\nTitle: First Command Financial Services\nContent: Wealth management and financial planning for military families. Tailored investment advisory, retirement strategies, and asset allocation.",
        "output": {
            "business_model": "D - Capital & advisory-led",
            "what_it_sells": "Capital & advisory",
            "build": "None",
            "institution_type": "Wealth Management / Brokerage",
            "sector_group": "Investing & wealth",
            "confidence": "High",
            "build_evidence": "Earns fees through financial planning, investment advisory, and wealth management services.",
            "what_they_do": "Financial advisory firm providing wealth management and investment planning."
        }
    }
]
