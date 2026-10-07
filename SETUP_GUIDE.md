# Pipecat Voice Agent - Setup & Demo Guide

## Prerequisites

You need API keys from:
1. **Deepgram** (Speech-to-Text) - https://console.deepgram.com/
2. **Cartesia** (Text-to-Speech) - https://www.cartesia.ai/
3. **Daily** (WebRTC Transport) - https://dashboard.daily.co/
4. **OpenRouter** (LLM) - https://openrouter.io/keys
   - Or OpenAI directly: https://platform.openai.com/api-keys

## Quick Start

### 1. Install Dependencies

```bash
cd server
uv sync
```

### 2. Configure Environment Variables

Edit `server/.env` with your API keys:

```bash
# Daily (Transport)
DAILY_API_KEY=your-daily-api-key

# Deepgram (STT)
DEEPGRAM_API_KEY=your-deepgram-api-key

# Cartesia (TTS)
CARTESIA_API_KEY=your-cartesia-api-key
CARTESIA_VOICE_ID=86e30c1d-714b-4074-a1f2-1cb6b552fb49

# OpenRouter LLM (Recommended for testing - cheaper than OpenAI)
OPENAI_API_KEY=your-openrouter-api-key
OPENAI_MODEL=gpt-4o-mini
LLM_BASE_URL=https://openrouter.io/api/v1
```

#### OpenRouter vs OpenAI

**Using OpenRouter:**
- More affordable for testing
- Access to multiple models: Claude, GPT, Llama, etc.
- Same OpenAI API format (fully compatible)
- Get key: https://openrouter.io/keys

**Using OpenAI directly:**
- Skip `LLM_BASE_URL` line (optional, defaults to OpenAI)
- Use your OpenAI API key
- Models like `gpt-4o`, `gpt-4-turbo`, etc.

### 3. Run the Bot

```bash
uv run bot.py
```

The server will start and print a URL like:
```
https://your-domain.daily.co/XXXXX
```

### 4. Test with Web Client

- Open the printed URL in your browser
- Click "Join" and allow microphone/speaker access
- Start talking! The agent will:
  1. Listen to your speech (Deepgram STT)
  2. Send it to LLM (OpenRouter)
  3. Speak the response back (Cartesia TTS)

## Architecture

```
Client (Browser)
     ↓
[WebRTC Transport - Daily]
     ↓
[STT - Deepgram] → [LLM - OpenRouter] → [TTS - Cartesia]
     ↓
Client (Audio Response)
```

## Troubleshooting

**"Invalid API key"** → Check `.env` file has correct keys and no extra spaces

**"Model not found"** → OpenRouter model names might differ from OpenAI
- Check available models: https://openrouter.io/docs/models

**"No audio output"** → 
- Check Cartesia API key and voice ID
- Verify browser audio permissions

**"STT not working"** →
- Check Deepgram API key
- Ensure microphone is enabled in browser

## Customization

### Change the System Prompt

Edit `bot.py` line ~76:
```python
system_instruction="Your custom instruction here..."
```

### Use a Different Model

```bash
# In .env
OPENAI_MODEL=claude-3.5-sonnet
# or
OPENAI_MODEL=meta-llama/llama-3-70b-instruct
```

### Add Voice ID from Cartesia

Get voice IDs: https://www.cartesia.ai/
```bash
CARTESIA_VOICE_ID=your-voice-id
```

## Next Steps

- Read the [Pipecat Docs](https://docs.pipecat.ai/)
- Explore [Pipecat Examples](https://github.com/pipecat-ai/pipecat-examples)
- Deploy to [Pipecat Cloud](https://docs.pipecat.ai/deployment/pipecat-cloud/introduction)
