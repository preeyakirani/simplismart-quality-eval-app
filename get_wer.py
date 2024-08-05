from transformers.models.whisper.english_normalizer import BasicTextNormalizer
# from indicnlp.normalize.indic_normalize import DevanagariNormalizer
from evaluate import load


def get_wer(prediction, truth):
    normalizer = BasicTextNormalizer(remove_diacritics=False, split_letters=False)
    normalised_prediction = normalizer(prediction)
    normalised_truth = normalizer(truth)

    # print("\n")
    # print("Prediction: " + normalised_prediction)
    # print("Truth: " + normalised_truth)
    # print("\n")
    
    wer = 100 * load("wer").compute(references=[normalised_truth], 
                                    predictions=[normalised_prediction])
    return round(wer, 3)