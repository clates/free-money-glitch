import json
import asyncio
import logging.handlers
import time
import os
import logging
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, SystemPrompt, Controller
from fetch_nasdaq_prices import get_next_business_days, fetch_earnings, get_next_monday
from pydantic import BaseModel
from typing import List

DATE_TO_START_ANALYSIS = "2025-02-17"
TOP_N_COMPANIES_BY_MARKET_CAP = 3
NUM_DAYS_LOOKAHEAD = 5
6
# Set up logging to both console and file
out_dir = "output"
log_time = time.strftime('%Y%m%d_%H%M')
log_dir = os.path.join(out_dir, log_time, "sentiment_logs")
gif_dir = os.path.join(out_dir, log_time, "gifs")
convo_dir = os.path.join(out_dir, log_time, "chat_logs")

os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
os.makedirs(gif_dir, exist_ok=True)  # Ensure the log directory exists
os.makedirs(convo_dir, exist_ok=True)  # Ensure the log directory exists

log_filename = os.path.join(log_dir, "output.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(log_filename)  # Output to file
    ]
)

# Define the structure of the expected JSON output


class SentimentResult(BaseModel):
    company_name: str
    company_ticker: str
    sentiment_score: int
    at_a_glance: str
    detailed_summary: str
    sources: List[str]

# Define custom system prompt

class MySystemPrompt(SystemPrompt):
    def important_rules(self) -> str:
        existing_rules = super().important_rules()
        new_rules = """
            9. USE GOOGLE FINANCE FINANCE RULE:
            - ALWAYS BEGIN BY SEARCHING FOR THE COMPANY ON Google with "GOOGLE FINANCE Finance" in the search query
            - This is USUALLY found at this URL. Note the pattern at the end includes the company ticker.
            - https://www.google.com/finance/quote/NVDA:\{company_ticker\}

            10. JSON RESPONSE RULE:
            - REMEMBER - your response must be valid JSON with the required fields. 
            - detailed_summary should contain a comprehensive analysis of the sentiment, including direct quotes from the articles.
            - at_a_glance should be a concise summary of the sentiment.

            11. SOURCES RULE:
            - sources should list all the URLs of the articles used in the analysis.
            """
        return f'{existing_rules}\n{new_rules}'


# Initialize the model
# llm = ChatOpenAI(model="gpt-4o")
# gemini is hella cheap
executor_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", api_key=SecretStr(os.getenv("GEMINI_API_KEY")))
planner_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21", api_key=SecretStr(os.getenv("GEMINI_API_KEY")))


async def analyze_company(company_name, company_ticker, market_cap, date, results):
    # Create a controller enforcing the SentimentAnalysis schema
    controller = Controller(output_model=SentimentResult)

    agent = Agent(
        task=f"Search for {company_name} ({company_ticker}) on Google Finance."
             f"Open and read the recent news articles until you are confidently able to determine the sentiment."
             f"Read at least five articles, but read more if you need to."
             f"If the company is not found, report that the company was not found."
             f"Determine a sentiment score (1-100) and provide a single at-a-glance evaluation."
             f"Additionally provide a more exhaustive followup summary paragraph citing insights derived from the articles."
             f"Include sources for the information found. Use direct quotes in your summary when possible.",
        planner_llm=executor_llm,
        llm=executor_llm,
        system_prompt_class=MySystemPrompt,
        save_conversation_path=os.path.join(
            convo_dir, f"conversation_{company_name}.json"),
        controller=controller,  # Enforce structured output
        generate_gif=os.path.join(gif_dir, f"{company_name}_sentiment.gif")
    )

    history = await agent.run()
    sentiment_result = history.final_result()

    if sentiment_result:
        try:
            parsed: SentimentResult = SentimentResult.model_validate_json(
                sentiment_result)
        except Exception as e:
            logging.error(
                f"❌ ERROR: Failed to parse sentiment data for {company_name} ({company_ticker})")
            logging.error(f"📅 Date: {date}")
            logging.error(f"⚠️ Raw response: {sentiment_result}")
            logging.error(f"🔴 Parsing error: {e}")
            results.append({
                "date_of_earnings_report": date,
                "company_name": company_name,
                "ticker": company_ticker,
                "market_cap": market_cap,
                # Ensure it's a dictionary
                "sentiment_result": "No data due to error in parsing the results",
            })
            return  # Skip this company if parsing fails

        logging.info(
            f"✅ RESULT ADDED: {parsed.company_name} ({parsed.company_ticker})")
        logging.info(f"📅 Date: {date}")
        logging.info(f"💰 Market Cap: {market_cap}")
        logging.info(f"📊 Sentiment Score: {parsed.sentiment_score}")
        logging.info(f"📝 Summary: {parsed.detailed_summary}")

        filename = f"sentiment_analysis_results_{company_name}.json"
        with open(os.path.join(log_dir, filename), "w") as f:
            f.write(parsed.model_dump_json(indent=4))

        results.append({
            "date_of_earnings_report": date,
            "company_name": parsed.company_name,
            "ticker": parsed.company_ticker,
            "market_cap": market_cap,
            "sentiment_score": parsed.sentiment_score,  # Ensure it's a dictionary
            "at_a_glance": parsed.at_a_glance,
            "summary": parsed.detailed_summary,
            "sources": parsed.sources
        })
    else:
        logging.error(
            f"❌ ERROR: No valid sentiment data for {company_name} ({company_ticker})")
        logging.error(f"📅 Date: {date}")
        logging.error("⚠️ Storing raw response for debugging.")
        results.append({
            "date_of_earnings_report": date,
            "company_name": company_name,
            "ticker": company_ticker,
            "market_cap": market_cap,
            "sentiment_result": "No sentiment data found.",  # Ensure it's a dictionary
            # "raw_result": str(history)  # Store the raw result for debugging
        })


async def main():
    start_time = time.time()  # Start timing
    all_results = []
    dates = get_next_business_days(NUM_DAYS_LOOKAHEAD, start_date=DATE_TO_START_ANALYSIS)

    logging.info(f"\n{'=' * 100}")
    logging.info("🚀 STARTING SENTIMENT ANALYSIS")
    logging.info(f"{'=' * 100}\n")

    for date in dates:
        logging.info(f"\n{'=' * 80}")
        logging.info(f"📅 FETCHING EARNINGS FOR {date}")
        logging.info(f"{'=' * 80}\n")

        earnings = fetch_earnings(date)

        # Sort by market cap and take the top 5
        top_earnings = sorted(earnings, key=lambda x: x.get(
            'marketCap', 0), reverse=True)[:TOP_N_COMPANIES_BY_MARKET_CAP]
        logging.info(json.dumps(top_earnings, indent=4))

        # Analyze each company and store results
        # Process companies **sequentially** instead of using asyncio.gather
        for company in top_earnings:
            try:
                await asyncio.wait_for(
                    analyze_company(
                        company['name'], company['symbol'], company['marketCap'], date, all_results),
                    # Fifteen minute Timeout
                    timeout=60*15
                )
            except Exception as e:
                logging.error(
                    f"❌ ERROR: Failed to analyze {company['name']} ({company['symbol']})")
                logging.error(f"🔴 Error: {e}")
            # Flush the log to ensure all messages are written out
            for handler in logging.getLogger().handlers:
                handler.flush()

        # fuck paraallelism
        # tasks = [analyze_company(company['name'], company['symbol'], company['marketCap'], date, all_results) for company in top_earnings]
        # await asyncio.gather(*tasks)

    # Save results to a file
    filename = f"sentiment_analysis_results.json"
    with open(os.path.join(out_dir, log_time, filename), "w") as f:
        json.dump(all_results, f, indent=4)

    logging.info(f"\n{'=' * 100}")
    logging.info(f"✅ ANALYSIS COMPLETE — RESULTS SAVED TO {filename}")
    logging.info(f"{'=' * 100}\n")

    # Display execution time
    execution_time = time.time() - start_time
    logging.info(f"⏱️ Total execution time: {execution_time:.2f} seconds")

# Run the async function
asyncio.run(main(), debug=True)
