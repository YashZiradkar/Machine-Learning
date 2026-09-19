from transformers import pipeline

MODEL = "anshy047/fake-news-detector-transformer"

classifier = pipeline(
    "text-classification",
    model = MODEL
)

def predict_news(text):

    result = classifier(text[:512])[0]

    return {
        "label": result["label"],
        "score": result["score"]
    }
