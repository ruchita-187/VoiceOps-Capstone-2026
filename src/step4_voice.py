import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient

# --------------------------------------------------
# PROJECT PATHS AND ENVIRONMENT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "voice"

sys.path.insert(0, str(SRC_DIR))

from step3_agent import process_message
from retrieve import retrieve

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("GNANI_API_KEY"):
    raise ValueError(
        "GNANI_API_KEY was not found. Add it to the .env file "
        "in the project root."
    )

# The Gnani SDK reads GNANI_API_KEY from the environment.
stt = GnaniSTTClient()
tts = GnaniTTSClient()

# --------------------------------------------------
# VOICE SETTINGS
# --------------------------------------------------

LANGUAGE_CODE = "en-IN"
TTS_VOICE = "Kaveri"
TTS_MODEL = "timbre-v2.5"

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg"
}


# --------------------------------------------------
# ONE VOICE TURN
# --------------------------------------------------

def voice_turn(audio_path, output_audio_path):
    audio_path = Path(audio_path)
    output_audio_path = Path(output_audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert customer audio to text using Gnani STT.
    stt_result = stt.transcribe(
        str(audio_path),
        language_code=LANGUAGE_CODE
    )

    heard = str(stt_result.get("transcript", "")).strip()

    # Step 2: Retrieve relevant company-document sections.
    # These are saved separately for the capstone batch requirement.
    retrieved_chunks = retrieve(heard) if heard else []

    # Step 3: Send the transcription to the existing Step 3 agent.
    # process_message() preserves its conversation and tool logic.
    if not heard:
        reply = "Sorry, I didn't catch that. Could you say it again?"
    else:
        reply = process_message(heard)

    # Step 4: Convert the agent's reply to speech using Gnani TTS.
    tts.synthesize(
        reply,
        voice=TTS_VOICE,
        model=TTS_MODEL,
        language=LANGUAGE_CODE,
        output_file=str(output_audio_path)
    )

    return heard, reply, retrieved_chunks, output_audio_path


# --------------------------------------------------
# SAVE OUTPUTS FOR ONE AUDIO FILE
# --------------------------------------------------

def process_audio_file(audio_path, output_folder):
    audio_path = Path(audio_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    reply_audio_path = output_folder / "reply.wav"
    reply_text_path = output_folder / "reply.txt"
    chunks_path = output_folder / "retrieved_chunks.json"

    heard, reply, retrieved_chunks, saved_audio = voice_turn(
        audio_path,
        reply_audio_path
    )

    reply_text_path.write_text(
        "Customer transcription:\n"
        f"{heard if heard else '[No speech detected]'}\n\n"
        "Assistant reply:\n"
        f"{reply}\n",
        encoding="utf-8"
    )

    with open(chunks_path, "w", encoding="utf-8") as file:
        json.dump(retrieved_chunks, file, ensure_ascii=False, indent=2)

    print(f"\nProcessed: {audio_path.name}")
    print(f"Customer transcription: {heard if heard else '[No speech detected]'}")
    print(f"Assistant reply: {reply}")
    print(f"Reply text saved to: {reply_text_path}")
    print(f"Reply audio saved to: {saved_audio}")
    print(f"Retrieved chunks saved to: {chunks_path}")


# --------------------------------------------------
# BATCH PROCESSING
# --------------------------------------------------

def process_audio_folder(input_folder):
    input_folder = Path(input_folder)

    if not input_folder.is_dir():
        raise NotADirectoryError(f"Audio folder not found: {input_folder}")

    audio_files = sorted(
        path for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(
            "No supported audio files found. Add WAV, MP3, M4A, FLAC "
            "or OGG files to the folder."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(audio_files)} audio file(s).")
    successful = 0
    failed = 0

    for number, audio_path in enumerate(audio_files, start=1):
        print("\n" + "=" * 60)
        print(f"Audio {number} of {len(audio_files)}: {audio_path.name}")
        print("=" * 60)

        output_folder = OUTPUT_DIR / audio_path.stem

        try:
            process_audio_file(audio_path, output_folder)
            successful += 1
        except Exception as error:
            failed += 1
            print(f"Failed to process {audio_path.name}: {error}")
            output_folder.mkdir(parents=True, exist_ok=True)
            (output_folder / "error.txt").write_text(
                str(error),
                encoding="utf-8"
            )

    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total audio files: {len(audio_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Outputs folder: {OUTPUT_DIR}")


# --------------------------------------------------
# COMMAND-LINE ENTRY POINT
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process one customer audio file or a folder of audio files "
            "through the ZentroMart voice agent."
        )
    )

    parser.add_argument(
        "input_path",
        help="Path to one audio file or a folder containing audio files."
    )

    args = parser.parse_args()
    input_path = Path(args.input_path)

    if input_path.is_file():
        output_folder = OUTPUT_DIR / input_path.stem
        try:
            process_audio_file(input_path, output_folder)
        except Exception as error:
            print(f"\nVoice processing failed: {error}")
            raise
    elif input_path.is_dir():
        process_audio_folder(input_path)
    else:
        raise FileNotFoundError(
            f"Input file or folder not found: {input_path}"
        )


if __name__ == "__main__":
    main()
