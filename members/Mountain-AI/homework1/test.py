#!/usr/bin/env python
import base64
from pathlib import Path

from jarvisbot import JarvisBot

j = JarvisBot(api_key="YourAPIKey")
messages = [
    {
        "content": "You are a helpful assistant.",
        "role": "system"
    },
    {
        "content": "What is the capital of France?",
        "role": "user"
    }
]
completion = j.chat.completions.create(
    messages=messages,
    model="llama2",
    response_format={"type": "json_object"},
    max_tokens=512,
    stream=False)

print(f"Prompt: {messages}")
print(completion.choices[0].message.content)


prompt = "A cute dog"
model = "StableDiffusion"
batch_count = 1
print(f"Prompt: {prompt}")
# Generate an image based on the prompt
response = j.images.generate(prompt=prompt, model=model, n=batch_count)
images = response.model_extra.get("images")
for index, image in enumerate(images):
    bs = base64.b64decode(image)
    with open(f"jarvisbot_sd_{index}.png", "wb") as f:
        f.write(bs)


speech_file_path = Path(__file__).parent / "speech.mp3"
# Create text-to-speech audio file
with j.audio.speech.with_streaming_response.create(
        model="vits",
        voice="female",
        input="the quick brown fox jumped over the lazy dogs",
) as response:
    response.stream_to_file(speech_file_path)


speech_file_path = Path(__file__).parent / "speech.mp3"
# Create transcription from audio file
transcription = j.audio.transcriptions.create(
    model="whisper",
    file=speech_file_path,
)
print(transcription.text)



