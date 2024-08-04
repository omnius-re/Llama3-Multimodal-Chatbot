# Discord Bot with YouTube and Image Processing

This Discord bot provides a variety of features, including chat interactions, summarizing YouTube video transcripts, fact-checking, and image descriptions. It uses the `discord.py` library for Discord interactions and integrates with various APIs for enhanced functionality.

## Features

- **Hello Command**: Responds with a friendly greeting.
- **Ask Command**: Sends a message to the Ollama chat model and returns the response.
- **Summarize Command**: Summarizes the last 10 messages in the channel.
- **TL;DR Command**: Fetches and summarizes YouTube video transcripts.
- **Fact-Check Command**: Fetches and fact-checks YouTube video transcripts.
- **Image Command**: Describes an image using OpenAI's API.

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>

2. **Install dependencies:**
    pip install discord.py ollama python-dotenv youtube-transcript-api tiktoken openai

3. **Create a .env file in the root directory with the following variables:**
    DISCORD_BOT_TOKEN=<your_discord_bot_token>
    CLIENT_ID=<your_openai_client_id>

4. **Run the bot:**
    python bot.py

## Commands

/hello: The bot responds with "Hello son :)".

/ask <message>: Sends the message to the Ollama chat model and returns the response.

/summarize: Summarizes the last 10 messages in the channel.

/tldr <url>: Fetches and summarizes the transcript of a YouTube video.

/factcheck <url>: Fetches and fact-checks the transcript of a YouTube video.

/image: Provides a description of the attached image using OpenAI's API.

## Environment Variables
    DISCORD_BOT_TOKEN: The token for your Discord bot.
    CLIENT_ID: Your OpenAI API key.

## Dependencies
    `discord.py`: A Python wrapper for the Discord API.
    `ollama`: A library for interacting with Ollama's chat models.
    `python-dotenv`: A library to read key-value pairs from a .env file.
    `youtube-transcript-api`: A library to fetch YouTube video transcripts.
    `tiktoken`: A library to encode text into tokens.
    `openai`: A library to interact with OpenAI's API.

## Troubleshooting
    Ensure all environment variables are correctly set.
    Verify that the required dependencies are installed.
    Check the bot's permissions on Discord to ensure it can read messages and send responses.