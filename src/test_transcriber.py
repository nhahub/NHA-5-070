from transcriber import transcribe


audio_path = "twin_data/test_audio/question1.m4a"

result = transcribe(audio_path)

print("\n===== TRANSCRIPTION =====")
print("Text:", result["text"])
print("Language:", result["language"])
print("Quality:", result["quality"])