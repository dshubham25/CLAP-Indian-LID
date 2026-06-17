# preprocess.py
import ssl
import urllib.request
import torch
import numpy as np
import librosa

# MAC FIX: Bypass SSL verification to allow PyTorch Hub download
ssl._create_default_https_context = ssl._create_unverified_context

# Safely attempt to load the Silero VAD model
try:
    print("Checking Silero Neural VAD model...")
    silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                         model='silero_vad',
                                         force_reload=False,
                                         trust_repo=True)
    get_speech_timestamps, _, _, _, collect_chunks = utils
    SILERO_LOADED = True
    print("Silero VAD successfully loaded!")
except Exception as e:
    print(f"\n[WARNING] Could not load Silero VAD: {e}")
    print("Falling back to standard Volume-based VAD (Librosa)...\n")
    SILERO_LOADED = False


def clean_and_format_audio(file_path, target_sr=48000, chunk_duration=5.0):
    """
    1. Neural VAD (Silero) bypass extraction
    2. Harmonic Source Separation (Removes percussive noise)
    3. Test-Time Augmentation (Pitch Shift & Time Stretch)
    """
    if SILERO_LOADED:
        wav_np, _ = librosa.load(file_path, sr=16000)
        wav = torch.from_numpy(wav_np).float()
        speech_timestamps = get_speech_timestamps(wav, silero_model, sampling_rate=16000)
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        if not speech_timestamps:
            raw_speech = audio
        else:
            speech_chunks = []
            for ts in speech_timestamps:
                start = int((ts['start'] / 16000) * target_sr)
                end = int((ts['end'] / 16000) * target_sr)
                speech_chunks.append(audio[start:end])
            raw_speech = np.concatenate(speech_chunks)
    else:
        audio, sr = librosa.load(file_path, sr=target_sr)
        intervals = librosa.effects.split(audio, top_db=25)
        if len(intervals) > 0:
            raw_speech = np.concatenate([audio[start:end] for start, end in intervals])
        else:
            raw_speech = audio

    # TECHNIQUE 1: HARMONIC SOURCE SEPARATION

    # Mathematically strips away sharp percussive background noise
    harmonic_speech, _ = librosa.effects.hpss(raw_speech, margin=1.2)

    # Base Temporal Chunks (Start, Middle, End)
    target_length = int(target_sr * chunk_duration)
    total_length = len(harmonic_speech)
    base_chunks = []
    
    if total_length < target_length:
        padding = target_length - total_length
        padded_chunk = np.pad(harmonic_speech, (0, padding), mode='constant')
        base_chunks.append(padded_chunk)
    else:
        base_chunks.append(harmonic_speech[:target_length])
        mid_start = (total_length - target_length) // 2
        base_chunks.append(harmonic_speech[mid_start : mid_start + target_length])
        end_start = total_length - target_length
        base_chunks.append(harmonic_speech[end_start:])


    # TECHNIQUE 3: TEST-TIME AUGMENTATION (TTA)

    # Create 3 versions of every chunk (Original, Pitched, Slowed)
    final_chunks = []
    for chunk in base_chunks:
        final_chunks.append(chunk) # 1. Original
        final_chunks.append(librosa.effects.pitch_shift(y=chunk, sr=target_sr, n_steps=1.5)) # 2. Pitch up slightly
        final_chunks.append(librosa.effects.time_stretch(y=chunk, rate=0.9)) # 3. Slow down by 10%
    
    return final_chunks, target_sr