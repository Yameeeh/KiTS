import os
import subprocess
import shutil
import librosa
import soundfile as sf
import taglib


def clear_folder(folder):
    """Clears all files in a folder."""
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared: {folder}")
    else:
        print(f"❌ Output folder '{folder}' does not exist.")


def convert_mp3_to_wav(input_file, output_file):
    """Converts an MP3 file to a WAV file using ffmpeg."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("❌ Error: ffmpeg is not installed or not found in system PATH.")
        return False

    try:
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le", output_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {input_file} to WAV: {e}")
        return False


def move_or_convert_audio_files(input_folder, temp_folder):
    """
    Step 0: Convert .mp3 files to .wav and move them to temp folder.
    If no MP3s exist, move existing .wav files to the temp folder.
    """
    os.makedirs(temp_folder, exist_ok=True)

    mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]
    wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".wav")]

    if mp3_files:
        print("MP3 files detected. Converting to WAV...")
        for filename in mp3_files:
            mp3_path = os.path.join(input_folder, filename)
            wav_path = os.path.join(temp_folder, filename.replace(".mp3", ".wav"))
            convert_mp3_to_wav(mp3_path, wav_path)
        print("Converted all .mp3s")
    elif wav_files:
        print("No MP3 files found. Copying existing WAV files to temp folder...")
        for filename in wav_files:
            old_path = os.path.join(input_folder, filename)
            new_path = os.path.join(temp_folder, filename)
            shutil.copy2(old_path, new_path)  # Copy instead of move
    else:
        print("❌ No audio files found in the input folder. Exiting.")
        return


def rename_wave_files(folder_path):
    """Renames .wav files in the folder to maintain their order and preserves their original names."""
    files = os.listdir(folder_path)
    wav_files = [f for f in files if f.lower().endswith('.wav')]

    # Sort the files to maintain order (assuming the names are numbers like 1.wav, 2.wav, etc.)
    wav_files.sort(key=lambda x: int(x.split('.')[0]))

    for index, wav_file in enumerate(wav_files, start=1):
        old_path = os.path.join(folder_path, wav_file)
        new_path = os.path.join(folder_path, f'{index}.wav')

        # Only rename if the filename is different
        if old_path != new_path:
            # Check if the new file already exists
            if os.path.exists(new_path):
                print(f"⚠️ Skipping rename: {new_path} already exists.")
                continue

            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} → {new_path}")


def clean_wav_metadata(input_file, cleaned_file):
    """Removes metadata from the WAV file."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("❌ Error: ffmpeg is not installed or not found in system PATH.")
        return False

    try:
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-c", "copy", "-map_metadata", "-1", cleaned_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error cleaning metadata from {input_file}: {e}")
        return False


def update_metadata(input_folder, output_folder):
    """Processes and updates metadata for all WAV files."""
    clear_folder(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wav_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".wav")])

    if not wav_files:
        print("❌ No .wav files found in temp folder.")
        return

    for index, filename in enumerate(wav_files, start=1):
        input_file = os.path.join(input_folder, filename)
        cleaned_file = os.path.join(input_folder, f"cleaned_{filename}")
        output_file = os.path.join(output_folder, f"{index}.wav")

        if not clean_wav_metadata(input_file, cleaned_file):
            print(f"⚠️ Skipping {filename} due to metadata cleaning error.")
            continue

        try:
            with taglib.File(cleaned_file, save_on_exit=True) as audio:
                audio.tags["TITLE"] = [str(index)]
                audio.tags["TRACKNUMBER"] = [str(index)]

            shutil.copy2(cleaned_file, output_file)

        except Exception as e:
            print(f"❌ Unexpected error: {e}")

        os.remove(cleaned_file)

    print("Successfully updated all metadata.")


def preprocess_and_save(input_path, output_path):
    """Trims silence and normalizes the audio."""
    for filename in os.listdir(input_path):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_path, filename)
            y, sr = librosa.load(filepath, sr=22050)

            # Trim silence
            trimmed_audio, _ = librosa.effects.trim(y, top_db=20)

            # Normalize audio
            normalized_audio = librosa.util.normalize(trimmed_audio)

            # Save processed .wav file
            output_filepath = os.path.join(output_path, filename)
            sf.write(output_filepath, normalized_audio, sr, subtype='PCM_16')

    print("✅ All WAV files preprocessed and saved.")


if __name__ == "__main__":
    input_folder = "processing/input"
    temp_folder = "processing/temp"
    temp2_folder = "processing/temp2"
    output_folder = "data/wavs"

    # Clear temp folders before processing
    clear_folder(temp_folder)
    clear_folder(temp2_folder)

    # Step 0: Move or convert audio files
    move_or_convert_audio_files(input_folder, temp_folder)

    # Step 1: Rename WAV files
    rename_wave_files(temp_folder)

    # Step 2: Preprocess and save to temp2 folder
    preprocess_and_save(temp_folder, temp2_folder)

    # Step 3: Update metadata and save to output folder
    update_metadata(temp2_folder, output_folder)

    # Step 4: Cleanup temp folders
    clear_folder(temp_folder)
    clear_folder(temp2_folder)

    print("🎉 All tasks completed successfully!")