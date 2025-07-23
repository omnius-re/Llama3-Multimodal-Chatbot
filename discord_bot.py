import os
import discord
from discord.ext import commands
import ollama
from dotenv import load_dotenv
import requests


from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import tiktoken

import openai
import base64
import json
from urllib.parse import urlparse

import uuid
import requests
import shutil


load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents= intents)


@bot.event
async def on_ready():
    print(f"Bot is ready as {bot.user.name}")


@bot.command(name="ask")
async def ask(ctx, *, message):
    print(message)
    print("=======")
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'system',
            'content' : 'You are a helpful and mature chatbot who responds to messages, sometimes using the catchphrase: I am a father before anything! Do not exceed 200 words in text generation.'
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
            'content': 'You are a helpful chatbot that summarizes the provided messages in bullet points concisely. Do not exceed 100 words in text generation.'
        }
    ])
    print(response['message']['content'])
    await ctx.send(response['message']['content'])

@bot.command(name="tldr")
async def yt_tldr(ctx, url: str):
    await ctx.send("Fetching transcript…")

    # Robustly extract the video ID
    try:
        if "youtube.com/watch?v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError("Unsupported YouTube URL format.")
    except Exception:
        await ctx.send("Could not parse the YouTube video ID. Please check the link.")
        return

    # Initialize YouTubeTranscriptApi
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id)  # uses updated API (v1.2.0+)
    except TranscriptsDisabled:
        return await ctx.send("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        return await ctx.send("No transcript found for this video.")
    except VideoUnavailable:
        return await ctx.send("Video is unavailable.")
    except Exception as e:
        return await ctx.send(f"Error fetching transcript: {e}")

    # Join transcript snippets
    full_transcript = " ".join([snippet.text for snippet in transcript])

    # Token count and chunking
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(full_transcript)
    num_tokens = len(tokens)
    print(f"Token count: {num_tokens}")

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
                        Please provide a summary for the following chunk of the YouTube video info:
                        1. Start with a high-level title for this chunk.
                        2. Provide 6-8 bullet points summarizing the key points in this chunk.
                        3. Start with the title of the chunk and then provide the summary in bullet points instead of using "Here's the summary of the transcript".
                        4. No need to use concluding remarks at the end.
                        5. Return the response in markdown format.
                        6. Add a divider at the end.

                        Chunk:
                        {chunk}
                        ''',
                    },
                ])
                summary = response.get('message', {}).get('content', '').strip()
                return summary if summary else "No summary available for this chunk."
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
                    Please provide a summary for the following chunk of the YouTube video info:
                    1. Create a title for the info
                    2. Return the response in markdown format.
                    3. Add a divider at the end.
                    {full_transcript}
                    ''',
                },
            ])
            summary = response.get('message', {}).get('content', '').strip()
            await ctx.send(summary if summary else "No summary available for this video.")
        except Exception as e:
            print(f"Error summarizing full transcript: {e}")
            await ctx.send("Error generating summary.")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import asyncio


# Load Vector Store with Domain Knowledge 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("knowledge_base", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context to answer the question. If context is insufficient, specify & use your existing knowledge to answer the question instead."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

output_parser = StrOutputParser()

# Initialize RAG LCEL Chain 
llm_model= ChatOpenAI(openai_api_key= os.getenv('api_key'),model_name="gpt-3.5-turbo")
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)

@bot.command(name="fc")
async def fc(ctx, url:str):
    await ctx.send("Fetching transcript…")

    # Robustly extract the video ID
    try:
        if "youtube.com/watch?v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError("Unsupported YouTube URL format.")
    except Exception:
        await ctx.send("Could not parse the YouTube video ID. Please check the link.")
        return

    # Initialize YouTubeTranscriptApi
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id)  # uses updated API (v1.2.0+)
    except TranscriptsDisabled:
        return await ctx.send("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        return await ctx.send("No transcript found for this video.")
    except VideoUnavailable:
        return await ctx.send("Video is unavailable.")
    except Exception as e:
        return await ctx.send(f"Error fetching transcript: {e}")

    # Prepare full transcript 
    full_transcript = " ".join([snippet.text for snippet in transcript])
    
    # Split transcript into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([full_transcript])

    # Expert Agentic Style Fact Checking
    async def analyze_chunk(chunk: Document, idx: int):
        question = f"""Please fact-check the following transcript segment as an expert appraiser:

        {chunk.page_content}

        1. Identify factual claims.
        2. Verify them using your knowledge.
        3. Return results in markdown.
        4. End with a confidence rating (High, Medium, Low).
        5. Do not hallucinate or make up data."""
        
        try:
            loop = asyncio.get_event_loop() #asynchronously loops through an event 
            result = await loop.run_in_executor(None, rag_chain.invoke, question) #loops through rag chain with each chunk of text
            return f"**Chunk {idx + 1}:**\n{result}\n---"
        except Exception as e:
            return f"❌ Error processing chunk {idx + 1}: {e}"

    # Process and send summaries
    await ctx.send(f"📄 Transcript split into {len(chunks)} chunks. Starting fact-checking...")

    for i, chunk in enumerate(chunks):
        summary = await analyze_chunk(chunk, i)
        await ctx.send(summary)

    await ctx.send("✅ Fact-checking completed.")

import vertexai
PROJECT_ID = os.getenv('PROJECT_ID')
location = os.getenv('location')

vertexai.init(project=PROJECT_ID, location=location)

from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    Image,
    Part
)



from langchain_google_vertexai import ChatVertexAI
from langchain.schema.messages import HumanMessage

image_directory = 'saved_images'
os.makedirs(image_directory, exist_ok=True)


@bot.command(name="im")
async def read_image(ctx, *, text_input: str = ""):
    try:
        # Retrieve the URL of the first attachment
        url = ctx.message.attachments[0].url
        print(f"Image URL: {url}")
    except IndexError:
        print("Error: No attachments")
        await ctx.send("No image detected.")
        return  # Exit the function if no attachment is found

    chat = ChatVertexAI(model_name="gemini-1.0-pro-vision")

    image_message = {
        "type": "image_url",
        "image_url": {"url": url},
    }
    text_message = {
        "type": "text",
        "text": f"What is this image? Return any texts on that image. {text_input} ",
    }

    try:
        message = HumanMessage(content=[text_message, image_message])
        response = chat([message])
        
        # Extract the 'content' from the response
        content = response.content

        await ctx.send(content)

    except Exception as e:
        print(f"Error: {e}")
        await ctx.send("An error occurred while processing the image.")

import fal_client

@bot.command(name="cook")
async def make_image(ctx, *, message):
    try:
        handler = fal_client.submit(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": message,
            },
        )

        result = handler.get()

        # Extract the image URL from the result
        image_url = result['images'][0]['url']
        print(image_url)
        await ctx.send(image_url)

    except Exception as e:
        print(f"Error: {e}")
        await ctx.send("An error occurred while generating the image.")





bot.run(os.getenv('DISCORD_BOT_TOKEN'))

