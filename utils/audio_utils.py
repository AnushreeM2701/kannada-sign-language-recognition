# utils/audio_utils.py

from gtts import gTTS
import os
import tempfile
import pygame

def play_audio(text: str, lang: str = 'kn'):
    """
    Generate and play audio for given text in the specified language.
    
    Parameters:
        text (str): The text to convert to speech.
        lang (str): Language code (e.g., 'kn' for Kannada, 'en' for English).
    """
    try:
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts = gTTS(text=text, lang=lang)
            tts.save(temp_audio.name)

            # Initialize mixer and play
            pygame.mixer.init()
            pygame.mixer.music.load(temp_audio.name)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                continue

            pygame.mixer.music.stop()
            pygame.mixer.quit()

        os.remove(temp_audio.name)

    except Exception as e:
        print(f"Error playing audio: {e}")
