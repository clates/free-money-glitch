import requests
import json
from datetime import datetime, timedelta
import numpy as np

def get_next_monday():
    today = datetime.today()
    days_until_monday = (7 - today.weekday()) % 7 or 7  # Ensures we always get the next Monday
    next_monday = today + timedelta(days=days_until_monday)
    return next_monday.strftime("%Y-%m-%d")

def get_next_business_days(n=5, start_date=None):
    # Default to today if no start_date provided
    if start_date is None:
        today = datetime.today()
    else:
        # Parse the start_date string
        today = datetime.strptime(start_date, "%Y-%m-%d")
        if today <= datetime.today():
            raise ValueError("Start date must be in the future")

    business_days = []
    while len(business_days) < n:
        today += timedelta(days=1)
        if today.weekday() < 5:  # Monday-Friday are business days
            business_days.append(today.strftime("%Y-%m-%d"))
    
    return business_days

def fetch_earnings(date: str):
    url = f"https://api.nasdaq.com/api/calendar/earnings?date={date}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses
        data = response.json()
        
        # Safety check for expected response structure
        if not data or "data" not in data or "rows" not in data["data"]:
            print("Unexpected response format")
            return []
        
        # Extract relevant earnings data
        earnings_rows = data["data"]["rows"]
        
        # Ensure market cap is parsed as a number for sorting
        for row in earnings_rows:
            market_cap_str = row.get("marketCap", "0").replace("$", "").replace(",", "").strip()
            try:
                row["marketCap"] = int(market_cap_str) if market_cap_str else 0  # Default to 0 if empty
            except ValueError:
                row["marketCap"] = 0  # Handle cases where market cap is still invalid
        
        # Sort by market cap in descending order and get top 5
        top_five = sorted(earnings_rows, key=lambda x: x["marketCap"], reverse=True)[:5]
        
        return top_five
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

if __name__ == "__main__":
    dates = get_next_business_days(5)
    for date in dates:
        print(f"Fetching earnings for {date}")
        top_earnings = fetch_earnings(date)
        print(json.dumps(top_earnings, indent=4))
