from transformers import pipeline
import matplotlib.pyplot as plt
def analyze_emotion(text):
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    emotion_scores = emotion_classifier(text)[0]
    emotions = {item['label']: item['score'] for item in emotion_scores}
    total_score = sum(emotions.values())
    emotion_percentages = {emotion: (score / total_score) * 100 for emotion, score in emotions.items()}
    return emotion_percentages
def plot_emotion_distribution(emotion_percentages):
    labels = list(emotion_percentages.keys())
    sizes = list(emotion_percentages.values())
    plt.figure(figsize=(7,7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Emotion Distribution')
    plt.show()
if __name__ == "__main__":
    text = input("Enter text to analyze emotions: ")
    emotion_percentages = analyze_emotion(text)
    print("Emotion distribution (in percentage):")
    for emotion, percentage in emotion_percentages.items():
        print(f"{emotion}: {percentage:.2f}%")
    plot_emotion_distribution(emotion_percentages)
