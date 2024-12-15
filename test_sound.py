import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Try using a different audio format (e.g., MP3)
try:
    pygame.mixer.music.load('sounds\detection_sound.mp3')  # Use an MP3 file instead
    pygame.mixer.music.play()
    print("Music loaded and playing")
except pygame.error as e:
    print(f"Error loading music: {e}")

pygame.time.wait(3000)  # Wait for 3 seconds to hear the sound
