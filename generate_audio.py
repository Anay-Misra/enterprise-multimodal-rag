from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os

# Define the script lines
script_lines = [
    ("SPEAKER 1", "Let's review Q2 performance. Revenue came in at $2.8 billion, which exceeded our guidance of $2.6 billion."),
    ("SPEAKER 2", "The outperformance was driven primarily by cloud services, which grew 28% year-over-year to $1.2 billion."),
    ("SPEAKER 3", "Our enterprise segment was particularly strong. We closed three deals over $10 million, including the Global Manufacturing Corp contract."),
    ("SPEAKER 1", "What about customer satisfaction metrics?"),
    ("SPEAKER 4", "NPS improved to 72, up from 68 last quarter. Support ticket resolution time decreased to 4.2 hours average.")
]

# Generate and save each part using gTTS (all male-style neutral voice as gTTS uses Google TTS)
file_paths = []
for i, (speaker, line) in enumerate(script_lines):
    tts = gTTS(text=line, lang='en', slow=False)
    filename = f"/mnt/data/line_{i+1}.mp3"
    tts.save(filename)
    file_paths.append(filename)

# Combine all parts into one MP3
combined = AudioSegment.empty()
for file_path in file_paths:
    audio = AudioSegment.from_mp3(file_path)
    combined += audio + AudioSegment.silent(duration=500)

# Export the combined audio
output_path = "/mnt/data/q2_performance_summary.mp3"
combined.export(output_path, format="mp3")

output_path
