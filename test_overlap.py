from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/overlapped-speech-detection",
   
)

output = pipeline("osd-test1.wav")

print("Detected overlapping regions:")
for segment in output.get_timeline():
    print(f"Overlap from {segment.start:.2f}s to {segment.end:.2f}s")
