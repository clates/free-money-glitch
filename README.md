# Free Money Glitch 🚀📈

An automated sentiment analysis tool that analyzes upcoming company earnings reports to help identify potential trading opportunities. This project combines web scraping, natural language processing, and LLM integrations to provide sentiment scores and comprehensive analysis for companies with upcoming earnings releases.

## 📋 Project Purpose

This repository is an exercise in:
- **Web scraping** techniques for financial data
- **LLM integration** capabilities in Python
- **Automated sentiment analysis** from news sources
- **Data visualization** for investment insights

The tool fetches upcoming earnings reports from NASDAQ, searches for recent news articles about each company, analyzes the sentiment using AI, and generates a "heat score" indicating potential buy or sell signals.

## ✨ Features

- 🔍 **Automated Earnings Data Fetching**: Retrieves upcoming earnings reports from NASDAQ API
- 📰 **News Aggregation**: Searches Google Finance for recent news articles about target companies
- 🤖 **AI-Powered Sentiment Analysis**: Uses LLM (Google Gemini or OpenAI) to analyze news sentiment
- 📊 **Sentiment Scoring**: Generates 1-100 sentiment scores for each company
- 🎨 **Visual Reports**: Creates infographic summaries with color-coded sentiment indicators
- 🎬 **Process Recording**: Generates GIFs showing the browser automation process
- 📝 **Detailed Logs**: Maintains comprehensive logs of all analysis steps

## 🛠️ Prerequisites

- Python 3.11 or higher
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv))
- Playwright browser automation
- API Keys:
  - Google Gemini API key (recommended - cost-effective)
  - OpenAI API key (optional alternative)

## 📦 Installation

```bash
# Create virtual environment with Python 3.11
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies
uv pip install browser-use

# Install Playwright browsers
playwright install

# Install Playwright system dependencies (may require sudo on Linux)
sudo playwright install-deps  # Linux
# playwright install-deps     # macOS/Windows

# Create necessary directories (handled automatically by scripts)
mkdir -p gifs logs sentiment_logs output
```

## ⚙️ Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```env
ANONYMIZED_TELEMETRY=false
GEMINI_API_KEY=your_gemini_api_key_here
#OPENAI_API_KEY=your_openai_key_here
```

You can obtain a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## 🚀 Usage

### 1. Fetch Earnings Data (Standalone)

```bash
python fetch_nasdaq_prices.py
```

This script fetches the top 5 companies by market cap with earnings reports in the next 5 business days.

### 2. Generate Sentiment Analysis

```bash
python gen_stock_sentiment.py
```

**Configuration options** (edit the constants at the top of the file):
- `DATE_TO_START_ANALYSIS`: Starting date for analysis (format: "YYYY-MM-DD")
- `TOP_N_COMPANIES_BY_MARKET_CAP`: Number of top companies to analyze per day (default: 3)
- `NUM_DAYS_LOOKAHEAD`: Number of business days to look ahead (default: 5)

**What it does**:
1. Fetches earnings data for the specified date range
2. For each company, launches a browser automation to:
   - Search Google Finance for the company
   - Read multiple recent news articles
   - Analyze sentiment using AI
   - Generate structured JSON output
3. Saves results, conversation logs, and GIFs

**Output locations**:
- `output/{timestamp}/sentiment_logs/` - Individual company analysis JSON files
- `output/{timestamp}/gifs/` - Browser automation GIFs
- `output/{timestamp}/chat_logs/` - Full conversation logs with the AI
- `output/{timestamp}/sentiment_analysis_results.json` - Aggregated results

### 3. Generate Infographic

```bash
python gen_infographic.py output/{timestamp}/sentiment_analysis_results.json
```

Creates a visual infographic (`sentiment_infographic.png`) with:
- Company names and tickers
- Sentiment scores (color-coded: green = bullish, red = bearish)
- "At a glance" summaries
- Market cap and earnings dates

## 📁 Project Structure

```
free-money-glitch/
├── fetch_nasdaq_prices.py      # Fetches earnings data from NASDAQ
├── gen_stock_sentiment.py      # Main sentiment analysis engine
├── gen_infographic.py          # Generates visual report
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
└── output/                     # Generated analysis results
    └── {timestamp}/
        ├── sentiment_logs/     # Individual company JSONs
        ├── gifs/              # Browser automation recordings
        ├── chat_logs/         # AI conversation logs
        └── sentiment_analysis_results.json
```

## 🔧 Technical Details

### Core Dependencies
- **browser-use**: Browser automation framework
- **langchain-google-genai**: Google Gemini LLM integration
- **playwright**: Web browser automation
- **pydantic**: Data validation and parsing
- **Pillow (PIL)**: Image generation for infographics
- **requests**: HTTP client for API calls

### Analysis Process
1. **Data Collection**: Queries NASDAQ API for earnings calendars
2. **Prioritization**: Sorts companies by market cap
3. **Research**: Browser automation navigates to Google Finance and news sources
4. **Analysis**: LLM reads articles and generates sentiment scores (1-100 scale)
5. **Validation**: Pydantic models ensure structured, type-safe outputs
6. **Visualization**: Generates color-coded infographics

### Sentiment Scoring
- **1-49**: Bearish sentiment (red tones)
- **50**: Neutral (gray)
- **51-100**: Bullish sentiment (green tones)

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is:
- NOT financial advice
- NOT a guarantee of investment performance
- An experimental project to explore web scraping and AI capabilities

**Always conduct your own research and consult with financial professionals before making investment decisions.**

## 📝 License

This is an open-source educational project. Use at your own risk.

---

**Note**: API costs apply for using Google Gemini or OpenAI. The project is configured to use Gemini 2.0 Flash by default for cost-effectiveness.