import sys
import os
import torch
import torchaudio
import shutil
import subprocess
import numpy as np
import librosa
from pydub import AudioSegment
import logging
from demucs.pretrained import get_model as demucs_get_model
from demucs.apply import apply_model
import whisper
SAMPLE_RATE = 44100
CLEANED_DIR = "speech_cleaned"
FINAL_DIR = "enhanced_output"
logging.basicConfig(filename='enhancer_errors.log', level=logging.ERROR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
demucs_model = demucs_get_model("htdemucs_6s").to(device).eval()
whisper_model = whisper.load_model("base", device=device)
def setup_directories():
    for d in [CLEANED_DIR, FINAL_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
def separate_with_demucs(input_path, output_dir="demucs_out"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    wav, sr = torchaudio.load(input_path)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    with torch.inference_mode(), torch.cuda.amp.autocast():
        sources = apply_model(demucs_model, wav[None], split=True, overlap=0.1, device=device)[0].cpu()
    for i, name in enumerate(demucs_model.sources):
        torchaudio.save(os.path.join(output_dir, f"{name}.wav"), sources[i], sr)
    return os.path.join(output_dir, "vocals.wav")
def detect_and_keep_speech(input_path, output_path):
    result = whisper_model.transcribe(input_path, fp16=False, verbose=False)
    if result["text"].strip():
        shutil.copy(input_path, output_path)
        return True
    return False
def bandpass_filter(input_path, output_path, low=100, high=8000):
    y, sr = librosa.load(input_path, sr=None)
    y_fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    mask = (freqs > low) & (freqs < high)
    y_filtered = np.fft.irfft(y_fft * mask).astype(np.float32)
    torchaudio.save(output_path, torch.from_numpy(y_filtered).unsqueeze(0), sr)
def process_full_audio(input_file):
    try:
        print("🔁 جاري معالجة الملف الصوتي بالكامل...")
        idx = 0
        vocal_path = separate_with_demucs(input_file)
        speech_path = os.path.join(CLEANED_DIR, f"speech_{idx}.wav")
        if detect_and_keep_speech(vocal_path, speech_path):
            audio = AudioSegment.from_wav(speech_path)
            if audio.dBFS > -35:
                filtered_path = os.path.join(FINAL_DIR, f"enhanced_{idx}.wav")
                bandpass_filter(speech_path, filtered_path)
            else:
                print(f"🔇 صوت منخفض جداً (dBFS = {audio.dBFS:.2f}) ➜ تحويله إلى صمت")
                silence = AudioSegment.silent(duration=len(audio))
                silence.export(os.path.join(FINAL_DIR, f"enhanced_{idx}.wav"), format="wav")
        else:
            print("⚠️ لا يوجد كلام ➜ تحويل الملف إلى صمت")
            duration_ms = len(AudioSegment.from_file(input_file))
            silence = AudioSegment.silent(duration=duration_ms)
            silence.export(os.path.join(FINAL_DIR, f"enhanced_{idx}.wav"), format="wav")
    except Exception as e:
        print(f"🔥 خطأ أثناء المعالجة: {str(e)}")
        logging.error(f"Error in full audio processing: {str(e)}")
def combine_chunks(output_file):
    files = sorted([os.path.join(FINAL_DIR, f) for f in os.listdir(FINAL_DIR) if f.endswith('.wav')])
    if not files:
        raise FileNotFoundError("❌ لا توجد ملفات صوتية للدمج")
    with open("inputs.txt", "w") as f:
        for file in files:
            f.write(f"file '{os.path.abspath(file)}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "inputs.txt", "-c", "copy", output_file], check=True)
    os.remove("inputs.txt")
if __name__ == "__main__":
    setup_directories()
    try:
        if len(sys.argv) != 3:
            print("❗ الاستخدام: python script.py input_audio.wav output.wav")
            sys.exit(1)
        process_full_audio(sys.argv[1])
        combine_chunks(sys.argv[2])
        print("✅ تم الانتهاء بنجاح!")
    except Exception as e:
        print(f"🔥 خطأ: {str(e)}")
        sys.exit(1)