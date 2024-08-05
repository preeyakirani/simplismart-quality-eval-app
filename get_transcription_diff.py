from dataclasses import dataclass
from pathlib import Path
from typing import List
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import re
import gradio as gr

import numpy as np
from minineedle import needle


@dataclass
class TextDiffRegion:
    reference_text: str
    compared_text: str
    match: bool


def text_diff(ref_text: str, compared: str) -> List[TextDiffRegion]:
    alignment = needle.NeedlemanWunsch(ref_text.lower().split(" "), compared.lower().split(" "))
    alignment.align()

    # Arrange
    regions = []
    for ref_word, compared_word in zip(*alignment.get_aligned_sequences()):
        regions.append(TextDiffRegion(
            ref_word if isinstance(ref_word, str) else "",
            compared_word if isinstance(compared_word, str) else "",
            match=(ref_word == compared_word)
        ))

    return regions

def get_highlighted_text(truth, text_to_compare):
    regions = text_diff(truth, text_to_compare)

    result = []
    for item in regions:
        if (item.match):
            result.append((item.reference_text, None))
        else:
            if (item.reference_text != "" and item.compared_text != ""):
                result.append((item.reference_text, "-"))
                result.append((item.compared_text, "+"))
            elif (item.reference_text == "" and item.compared_text != ""):
                result.append((item.compared_text, "+"))
            elif (item.reference_text != "" and item.compared_text == ""):
                result.append((item.reference_text, "-"))
        
    return result

def get_highlighted_text_capitalised(truth, prediction):
    truth = truth.strip()
    prediction = prediction.strip()

    regions = text_diff(truth, prediction)

    truth_arr = re.split(" ", truth)
    prediction_arr = re.split(" ", prediction)
    truth_arr.append("")
    prediction_arr.append("")

    count_truth = 0
    count_pred = 0;

    result = []
    
    for item in regions:
        if (item.reference_text == "" and item.compared_text == ""):
            break;
        if (item.match):
            result.append((prediction_arr[count_pred], None))
            count_truth += 1
            count_pred += 1
        else:
            if (item.reference_text != "" and item.compared_text != ""):
                result.append((truth_arr[count_truth], "-"))
                count_truth += 1
                result.append((prediction_arr[count_pred], "+"))
                count_truth += 1
            elif (item.reference_text == "" and item.compared_text != ""):
                result.append((prediction_arr[count_pred], "+"))
                count_pred += 1
            elif (item.reference_text != "" and item.compared_text == ""):
                result.append((truth_arr[count_truth], "-"))
                count_truth += 1
        
    return result



def colorify(truth, prediction):
    truth = truth.strip()
    prediction = prediction.strip()

    diff = text_diff(truth, prediction)

    truth_arr = re.split(" ", truth)
    prediction_arr = re.split(" ", prediction)
    truth_arr.append("")
    prediction_arr.append("")

    count_truth = 0
    count_pred = 0;

    colorified_transcript = ""

    for obj in diff:
        if (obj.reference_text == "" and obj.compared_text == ""):
            break;
        if (obj.match == False):
            if (obj.reference_text != "" and obj.compared_text != ""):
                colorified_transcript += ("[color=red]" + truth_arr[count_truth] + "[/color] ")
                colorified_transcript += ("[color=green]" + prediction_arr[count_pred] + "[/color] ")
                count_truth += 1
                count_pred += 1
            elif (obj.reference_text != "" and obj.compared_text == ""):
                colorified_transcript += ("[color=red]" + truth_arr[count_truth] + "[/color] ")
                count_truth += 1
            elif (obj.reference_text == "" and obj.compared_text != ""):
                colorified_transcript += ("[color=green]" + prediction_arr[count_pred] + "[/color] ")
                count_pred += 1
        else:
            colorified_transcript += (prediction_arr[count_pred]) + " "
            count_truth += 1
            count_pred += 1

    # colorified_transcript += "</div>"
    return colorified_transcript

#print(get_highlighted_text_capitalised("This is correct.", "Or this is Correct."))

# def colorify(truth, prediction, service):
#     truth = truth.strip()
#     prediction = prediction.strip()

#     diff = text_diff(truth, prediction)

#     truth_arr = re.split(" ", truth)
#     prediction_arr = re.split(" ", prediction)
#     truth_arr.append("")
#     prediction_arr.append("")

#     count_truth = 0
#     count_pred = 0;

#     colorified_transcript = f"""
#     <div style=
#         "background-color:#1f2937;
#         border:#374151 solid 1px;
#         border-radius:8px;
#         padding:10px;
#         width:71.43%">

#         <p style=
#                 "margin:0px 0px 5px 5px">
#         {service}
#         </p>

#         <div style=
#             "line-height:1.5; 
#             background-color:#1f2937; 
#             border:#374151 solid 1px; 
#             border-radius:8px; 
#             padding:10px;
#             width:auto"
#         >
#     """

#     for obj in diff:
#         if (obj.reference_text == "" and obj.compared_text == ""):
#             break;
#         if (obj.match == False):
#             if (obj.reference_text != "" and obj.compared_text != ""):
#                 colorified_transcript += ('<span style="background-color:#b50d1e; padding:1px; border-radius:2px">' + truth_arr[count_truth] + "</span> ")
#                 colorified_transcript += ('<span style="background-color:#34821d; padding:1px; border-radius:2px">' + prediction_arr[count_pred] + "</span> ")
#                 count_truth += 1
#                 count_pred += 1
#             elif (obj.reference_text != "" and obj.compared_text == ""):
#                 colorified_transcript += ('<span style="background-color:#b50d1e; padding:1px; border-radius:2px">' + truth_arr[count_truth] + "</span> ")
#                 count_truth += 1
#             elif (obj.reference_text == "" and obj.compared_text != ""):
#                 colorified_transcript += ('<span style="background-color:#34821d; padding:1px; border-radius:2px">' + prediction_arr[count_pred] + "</span> ")
#                 count_pred += 1
#         else:
#             colorified_transcript += (prediction_arr[count_pred]) + " "
#             count_truth += 1
#             count_pred += 1

#     colorified_transcript += "</div> </div>"
#     return colorified_transcript

# with gr.Blocks() as demo:
#     gr.HTML(value=colorify("This is right. This is right.", "This is write. This is write.", "Simplismart"))

# demo.launch()