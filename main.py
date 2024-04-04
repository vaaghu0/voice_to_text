import whisperx


def main(audio_file):

    device = "cuda"
    # audio_file = "Road.wav"
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
    model = whisperx.load_model("large-v2", device="cuda", compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language="en")
    print(result["segments"])  # before alignment


if __name__ == "__main__":
    audio_file = input("Enter audio file path: ")
    main(audio_file=audio_file)
