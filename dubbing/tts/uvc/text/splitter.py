from .cleaners import uzbek_cleaners
from pydub.silence import split_on_silence
from pydub import AudioSegment

def splited_words(text, valium=5):
    all_sentence = []  
    text = text.split()
    text_chunks = [" ".join(text[i:i+valium]) for i in range(0, len(text), valium)]

    for line in text_chunks:  
        all_sentence.append(line+" .")
    return all_sentence

def split_sentence(text):
    text = uzbek_cleaners(text=text)  
    text = splited_words(text=text) 
    return text

def remove_silence(audio_path, output_file, min_silence_len=200, silence_thred=-40):
    sound = AudioSegment.from_wav(audio_path)
    audio_chunks = split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thred,
    )
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(output_file, format="wav")
    
