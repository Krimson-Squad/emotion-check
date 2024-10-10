from transformers import pipeline
from colorama import Fore, Style, init
init(autoreset=True)
def analyze_emotion(text):
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    emotion_scores = emotion_classifier(text)[0]    
    emotions = {item['label']: item['score'] for item in emotion_scores}
    total_score = sum(emotions.values())
    emotion_percentages = {emotion: (score / total_score) * 100 for emotion, score in emotions.items()}   
    return emotion_percentages
def print_ascii_pie(emotion_percentages):
    total_length = 40  
    print("\nEmotion Distribution (ASCII Chart):\n")
    color_mapping = {
        "joy": Fore.GREEN,
        "anger": Fore.RED,
        "disgust": Fore.MAGENTA,
        "fear": Fore.YELLOW,
        "neutral": Fore.CYAN,
        "sadness": Fore.BLUE,
        "surprise": Fore.WHITE,
    }
    for emotion, percentage in emotion_percentages.items():
        num_chars = int(total_length * (percentage / 100))
        color = color_mapping.get(emotion, Fore.WHITE)  
        print(f"{color}{emotion:10}: {'#' * num_chars} ({percentage:.2f}%)")
    print(Style.RESET_ALL + "\n" + "-" * 60)
if __name__ == "__main__":
    text = input("Enter text to analyze emotions: ")
    emotion_percentages = analyze_emotion(text)
    print_ascii_pie(emotion_percentages)