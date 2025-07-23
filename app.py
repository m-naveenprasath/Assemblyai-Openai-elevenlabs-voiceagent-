import os
import traceback
import openai
import assemblyai as aai
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile

# Load API keys from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

# Home loan expert prompt
SYSTEM_PROMPT = (
    "You are a skilled and friendly home loan expert. Answer questions clearly, helpfully, "
    "and professionally. If a user asks about eligibility, interest rates, documents, or steps, "
    "explain as a knowledgeable human advisor."
)

# 🎤 Record user question
def record_audio(duration=5, fs=16000):
    print("🎤 Speak your home loan question (recording 5 sec)...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(tmpfile.name, fs, audio_data)
    return tmpfile.name

# 📝 Transcribe audio with AssemblyAI
def transcribe_audio(audio_path):
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path)
        return transcript.text
    except Exception as e:
        print("❌ Error during transcription:", e)
        traceback.print_exc()
        return ""

# 🧠 Get GPT response
def get_home_loan_answer(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ Error from OpenAI:", e)
        traceback.print_exc()
        return "Sorry, I couldn't process that."

# 🔊 Speak using ElevenLabs
def speak_response(text):
    try:
        print("🧠 Generating voice...")
        client = ElevenLabs(api_key=elevenlabs_api_key)
        response = client.voices.get_all()
        voice_id = None
        for voice in response.voices:
            if voice.name.lower() == "aria":
                voice_id = voice.voice_id
                break
        if not voice_id:
            raise Exception("Voice 'Aria' not found.")
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2"
        )
        print("🔊 Speaking...")
        play(audio)
    except Exception as e:
        print("❌ Error in speak_response:", e)
        traceback.print_exc()


# 🔁 Run the full flow
def run_conversation():
    audio_path = record_audio()
    question = transcribe_audio(audio_path)

    if question:
        print(f"🧾 You asked: {question}")
        answer = get_home_loan_answer(question)
        print(f"🤖 Expert says: {answer}")
        speak_response(answer)
    else:
        print("⚠️ No valid question detected.")

# 🚀 Main loop
if __name__ == "__main__":
    while True:
        run_conversation()
        again = input("🔁 Ask another question? (y/n): ").strip().lower()
        if again != "y":
            print("👋 Goodbye!")
            break
