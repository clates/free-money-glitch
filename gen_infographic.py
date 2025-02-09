import sys
import json
import time
from typing import List
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont


class SentimentResult(BaseModel):
    company_name: str
    at_a_glance: str
    date_of_earnings_report: str
    market_cap: int
    sentiment_score: int
    sources: List[str]
    summary: str
    ticker: str


def load_sentiment_data(file_path: str) -> List[SentimentResult]:
    """Load and parse sentiment data from JSON."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return [SentimentResult(**item) for item in data]


def sentiment_color(score: int) -> tuple:
    """Return an RGB color based on sentiment score (0-100 scale, adjusted)."""

    # Adjust score: Anything below 50 is treated as 50 for color calculation
    adjusted_score = max(50, score)  # Ensure score is at least 50

    # Scale the adjusted score to a -50 to +50 range for color mapping
    scaled_score = adjusted_score - 50

    if scaled_score > 0:
        # Adjust multiplier for desired green intensity
        green = min(255, scaled_score * 4)
        # Adjust multiplier for desired red intensity
        red = min(255, abs(50-scaled_score) * 4)
        return (red, green, 0)
    else:
        return (128, 128, 128)  # Neutral (Gray)


def wrap_text(text, font, max_width):
    """Wrap text to fit within a specified width."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (word if current_line == "" else " ") + word
        # Use getbbox() for more accurate width calculation
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]  # Calculate width from bounding box
        if width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)  # Add the last line
    return "\n".join(lines)


def generate_infographic(sentiment_results: List[SentimentResult], output_file="sentiment_infographic.png"):
    """Generate an infographic from sentiment data."""
    sentiment_results.sort(key=lambda x: x.sentiment_score,
                           reverse=True)  # Sort by sentiment score descending

    # Adjust height based on entries
    width, height = 2000, 200 + len(sentiment_results) * 350
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Load fonts, with fallback
    try:
        font_title = ImageFont.truetype("arial.ttf", 60)  # Reduced title size
        font_large = ImageFont.truetype(
            "arial.ttf", 45)  # Reduced "at a glance" size
        # Reduced market cap and earnings size
        font_medium = ImageFont.truetype("arial.ttf", 35)
        font_small = ImageFont.truetype(
            "arial.ttf", 30)  # Reduced summary size
        # Added font for sources if needed.
        font_verysmall = ImageFont.truetype("arial.ttf", 25)
    except IOError:
        font_title = font_large = font_medium = font_small = font_verysmall = ImageFont.load_default()

    # Header
    draw.text((50, 50), f'FMG - Sentiment Analysis - {time.strftime("%b %d")} ',
              fill="black", font=font_title)

    y_offset = 150
    bg_colors = [(240, 240, 240), (220, 220, 220)]  # Alternating row colors

    for i, result in enumerate(sentiment_results):
        row_color = bg_colors[i % 2]  # Alternate background colors
        draw.rectangle(
            [(0, y_offset), (width, y_offset + 350)], fill=row_color)

        sentiment_text_color = sentiment_color(result.sentiment_score)
        # Company Name, Ticker, Score
        draw.text((50, y_offset + 20),
                  f"{result.company_name} ({result.ticker})", fill="black", font=font_title)
        sentiment_text_color = sentiment_color(result.sentiment_score)
        draw.text((1600, y_offset + 20),
                  f"Score: {result.sentiment_score}", fill=sentiment_text_color, font=font_title)

        # At a Glance (with wrapping, extended width, italicized, slightly lighter color)
        at_a_glance_wrapped = wrap_text(result.at_a_glance, font_large, 1800)
        # Try italic font, if not default to a lighter gray
        try:
            font_large_italic = ImageFont.truetype(
                "Arial-ItalicMT.ttf", 30)  # Or "ariali.ttf"
            draw.text((50, y_offset + 90), at_a_glance_wrapped,
                      fill=(80, 80, 80), font=font_large_italic)
        except IOError:
            draw.text((50, y_offset + 90), at_a_glance_wrapped,
                      fill=(80, 80, 80), font=font_large)

        # Market Cap & Earnings Date (moved down)
        # Adjusted position
        draw.text((50, y_offset + 275),
                  f"Market Cap: ${result.market_cap:,}", fill="black", font=font_medium)
        # Adjusted position
        draw.text((700, y_offset + 275),
                  f"Earnings Date: {result.date_of_earnings_report}", fill="black", font=font_medium)

        # Delimiter line (moved down)
        # draw.line([(50, y_offset + 200), (width - 50, y_offset + 200)],
        #          fill="black", width=3)  # Adjusted position

        y_offset += 350  # Adjust spacing

    image.save(output_file)
    print(f"Infographic saved as {output_file}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py sentiment_data.json")
        sys.exit(1)

    input_file = sys.argv[1]
    sentiment_data = load_sentiment_data(input_file)
    generate_infographic(sentiment_data)


if __name__ == "__main__":
    main()
