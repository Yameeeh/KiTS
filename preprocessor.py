import os
import subprocess
import shutil
import librosa
import soundfile as sf
import taglib


def clear_folder(folder):
    """
    Clears all files in a folder.
    """
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    else:
        print(f"❌ Output folder '{folder}' does not exist.")


def rename_wave_files(folder_path):
    """
    Renames the .wav files in the folder sequentially.
    """
    files = os.listdir(folder_path)
    wav_files = [f for f in files if f.lower().endswith('.wav')]

    print(f'Folder path: {folder_path}')
    print(f'All files in the folder: {files}')
    print(f'Wave files found: {len(wav_files)}')

    for index, wav_file in enumerate(wav_files, start=1):
        old_path = os.path.join(folder_path, wav_file)
        new_path = os.path.join(folder_path, f'{index}.wav')
        os.rename(old_path, new_path)
        print(f'Renamed {old_path} to {new_path}')


def clean_wav_metadata(input_file, cleaned_file):
    """
    Removes any existing metadata (including ID3 tags) from the WAV file.
    """
    # Check if ffmpeg is available
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
    clear_folder(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wav_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".wav")])

    if not wav_files:
        print("No .wav files found in the temp folder.")
        return

    for index, filename in enumerate(wav_files, start=1):
        input_file = os.path.join(input_folder, filename)
        cleaned_file = os.path.join(input_folder, f"cleaned_{filename}")
        output_file = os.path.join(output_folder, f"{index}.wav")

        # Remove any existing metadata
        if not clean_wav_metadata(input_file, cleaned_file):
            print(f"❌ Skipping {filename} due to metadata cleaning error.")
            continue

        try:
            # Open the cleaned WAV file using taglib
            with taglib.File(cleaned_file, save_on_exit=True) as audio:
                # Set the title and track number using lists
                audio.tags["TITLE"] = [str(index)]
                audio.tags["TRACKNUMBER"] = [str(index)]

            # Copy the final file to the output folder
            shutil.copy2(cleaned_file, output_file)

        except Exception as e:
            print(f"❌ Unexpected error: {e}")

        # Remove the cleaned file after processing
        os.remove(cleaned_file)
    print("Successfully updated all metadata")


def preprocess_and_save(input_path, output_path):
    """
    Preprocess the audio files: trim silence and normalize the audio.
    """
    for filename in os.listdir(input_path):
        if filename.endswith(".wav"):
            # Load the .wav file
            filepath = os.path.join(input_path, filename)
            y, sr = librosa.load(filepath, sr=22050)

            # Trim silence
            trimmed_audio, _ = librosa.effects.trim(y, top_db=20)

            # Normalize audio
            normalized_audio = librosa.util.normalize(trimmed_audio)

            # Save processed .wav file to the output folder
            output_filepath = os.path.join(output_path, filename)
            sf.write(output_filepath, normalized_audio, sr, subtype='PCM_16')

    print("All .wav files have been preprocessed and saved to the output folder.")


if __name__ == "__main__":
    input_folder = "inputWavs"
    temp_folder = "temp"
    output_folder = "wavs"
    
    # Clear the temp folder
    clear_folder(temp_folder)

    # Step 1: Rename .wav files in the input folder
    rename_wave_files(input_folder)

    # Step 2: Preprocess and save the .wav files to the output folder
    preprocess_and_save(input_folder, temp_folder)

    # Step 3: Update metadata for the .wav files in the output folder
    update_metadata(temp_folder, output_folder)

    # Step 4: Clear Temp Folder
    clear_folder(temp_folder)

    print("All tasks completed successfully!")
