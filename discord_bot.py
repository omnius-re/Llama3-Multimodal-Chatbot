import os
import discord
from discord.ext import commands
import ollama
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
import tiktoken


load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents= intents)

@bot.event
async def on_ready():
    print(f"Bot is ready as {bot.user.name}")

@bot.command(name="hello")
async def hello(ctx):
    await ctx.send("Hello son :)") 

@bot.command(name="ask")
async def ask(ctx, *, message):
    print(message)
    print("=======")
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'system',
            'content' : 'You are Lebron James. You are a helpful and mature chatbot who responds to messages, sometimes using the catchphrase: I am a father before anything! Do not exceed 100 words in text generation.'
        },
        {
            'role': 'user',
            'content': message,
        },
    ])
    print(response['message']['content'])
    await ctx.send(response['message']['content'])

@bot.command(name="summarize")
async def summarize(ctx):

    msgs = [message.content async for message in ctx.channel.history(limit=10)]

    summarize_prompt = f"""
        Summarize the following messages delimited by 3 backticks:
        ```
        {msgs}
        ```
        """
    
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': 'You are Lebron James. You are a helpful chatbot that summarizes the provided messages in bullet points concisely. Do not exceed 100 words in text generation.'
        }
    ])
    print(response['message']['content'])
    await ctx.send(response['message']['content'])

@bot.command(name="tldr")
async def yt_tldr(ctx, url):
    await ctx.send("Fetching and summarizing the YouTube video...")

    # Validate the URL format
    if "v=" not in url:
        await ctx.send("Invalid YouTube URL format. Please provide a valid video URL.")
        return

    try:
        video_id = url.split("v=")[1].split("&")[0]  # Handle cases where additional parameters are present
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        await ctx.send(f"Error fetching transcript: {e}")
        return

    full_transcript = " ".join([item['text'] for item in transcript_list])

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(full_transcript)
    num_tokens = len(tokens)

    print(num_tokens)
    chunk_size = 7000

    if num_tokens > chunk_size:
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size
        chunks = [full_transcript[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

        async def process_chunk(chunk, chunk_num):
            await ctx.send(f"Extracting summary of chunk {chunk_num} of {num_chunks}...")

            try:
                response = ollama.chat(model='llama3', messages=[
                    {
                        'role': 'user',
                        'content': '''You are a helpful assistant. You are a helpful chatbot that provides a concise 
                        summary of the Youtube video in bullet points. Do not exceed 100 words in text generation.'''
                    },
                    {
                        'role': 'user',
                        'content': f'''
                        Please provide a summary for the following chunk of the YouTube video transcript:
                        {chunk}
                        ''',
                    },
                ])
                print(f"Response for chunk {chunk_num}: {response}")

                summary = response.get('message', {}).get('content', '').strip()
                if not summary:
                    summary = "No summary available for this chunk."

                return summary
            except Exception as e:
                print(f"Error processing chunk {chunk_num}: {e}")
                return "Error generating summary."

        for i, chunk in enumerate(chunks, start=1):
            try:
                summary = await process_chunk(chunk, i)
                await ctx.send(summary)
            except Exception as e:
                print(f"Error sending summary for chunk {i}: {e}")
                await ctx.send("Error sending summary.")
    else:
        try:
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': '''You are a helpful assistant. You are a helpful chatbot that provides a concise 
                    summary of the Youtube video in bullet points. Do not exceed 100 words in text generation.'''
                },
                {
                    'role': 'user',
                    'content': f'''
                    Please provide a summary for the following chunk of the YouTube video transcript:
                    {full_transcript}
                    ''',
                },
            ])
            print(f"Response for full transcript: {response}")

            summary = response.get('message', {}).get('content', '').strip()
            if not summary:
                summary = "No summary available for this video."

            await ctx.send(summary)
        except Exception as e:
            print(f"Error summarizing full transcript: {e}")
            await ctx.send("Error generating summary.")



bot.run(os.getenv("DISCORD_BOT_TOKEN"))

