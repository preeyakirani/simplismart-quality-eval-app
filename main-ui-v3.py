import gradio as gr
from gradio_rich_textbox import RichTextbox
import time
import json
import re
import traceback

from get_deepgram_tr import *
from get_simplismart_tr import *
from get_assembly_ai_tr import *
from get_rev_ai_tr import *
from get_whisper_tr import *
from get_aws_tr import *

from get_wer import *
from get_transcription_diff import *


########################## FILE TRANSCRIBE ##########################

def get_lang_code(language):
    if (language == "None (Detection)"):
        return None
    elif (language == "English"):
        return "en"
    elif (language == "Hindi"):
        return "hi"
    elif (language == "Spanish"):
        return "es"


def file_transcribe_simplismart(audio_file, 
                                actual_tr, 
                                task, 
                                language, 
                                initial_prompt, 
                                beam_size, 
                                best_of, 
                                patience, 
                                vad_onset, 
                                vad_offset, 
                                min_duration_on, 
                                min_duration_off, 
                                pad_onset, 
                                pad_offset):
    
    simplismart_res, time_taken = get_transcription_simplismart(audio_file, 
                                                                task, 
                                                                get_lang_code(language), 
                                                                initial_prompt, 
                                                                beam_size, 
                                                                best_of, 
                                                                patience, 
                                                                vad_onset, 
                                                                vad_offset, 
                                                                min_duration_on, 
                                                                min_duration_off, 
                                                                pad_onset, 
                                                                pad_offset)
    # predictions
    simplismart_prediction = ""
    for item in simplismart_res["transcription"]:
        simplismart_prediction += item["text"]

    if (actual_tr != ""):
        colored_simplismart_prediction = colorify(actual_tr, simplismart_prediction)
        simplismart_wer = get_wer(simplismart_prediction, actual_tr)
    else:
        colored_simplismart_prediction = colorify(simplismart_prediction, simplismart_prediction)
        simplismart_wer = "NA"
    
    #highlighted_simplismart_prediction = get_highlighted_text_capitalised(actual_tr, simplismart_prediction)

    return [json.dumps(simplismart_res), colored_simplismart_prediction, simplismart_wer, time_taken,
            gr.Button(value="Submit", variant="primary", interactive=False)]


def file_transcribe_deepgram(audio_file, actual_tr, task, language):
    if (task == "translate"):
        return ["", "{TRANSLATION NOT SUPPORTED NATIVELY}", "NA", 0.00,
                gr.Button(value="Submit", variant="primary", interactive=False)]
    else:
        deepgram_res, time_taken = get_transcription_deepgram(audio_file, get_lang_code(language))
        deepgram_prediction = deepgram_res["results"]["channels"][0]["alternatives"][0]["transcript"]

        if (actual_tr != ""):
            colored_deepgram_prediction = colorify(actual_tr, deepgram_prediction)
            deepgram_wer = get_wer(deepgram_prediction, actual_tr)
        else:
            colored_deepgram_prediction = colorify(deepgram_prediction, deepgram_prediction)
            deepgram_wer = "NA"
        
        return [deepgram_res, colored_deepgram_prediction, deepgram_wer, time_taken,
                gr.Button(value="Submit", variant="primary", interactive=False)]


def file_transcribe_assemblyai(audio_file, actual_tr, task, language):
    if (task == "translate"):
        return ["", "{TRANSLATION NOT SUPPORTED NATIVELY}", "NA", 0.00,
                gr.Button(value="Submit", variant="primary", interactive=False)]
    else:
        assembly_res, time_taken = get_transcription_assemblyai(audio_file, get_lang_code(language))
        assembly_prediction = assembly_res.text

        if (actual_tr != ""):
            colored_assembly_prediction = colorify(actual_tr, assembly_prediction)
            assembly_wer = get_wer(assembly_prediction, actual_tr)
        else:
            colored_assembly_prediction = colorify(assembly_prediction, assembly_prediction)
            assembly_wer = "NA"


        response_words = [];
        for word in assembly_res.words:
            response_words.append({
                "start": word.start/1000,
                "end": word.end/1000,
                "text": word.text
            })

        return [response_words, colored_assembly_prediction, assembly_wer, time_taken,
                gr.Button(value="Submit", variant="primary", interactive=False)]


def file_transcribe_rev_ai(audio_file, actual_tr, task, language):
    # print(audio_file)
    # print(task)
    # print(get_lang_code(language))
    rev_res_text, rev_res, time_taken = get_transcription_rev_ai(audio_file, task, get_lang_code(language))
    rev_prediction = (re.split("    |\n", rev_res_text))[2].strip()

    if (actual_tr != ""):
        colored_rev_prediction = colorify(actual_tr, rev_prediction)
        rev_wer = get_wer(rev_prediction, actual_tr)
    else:
        colored_rev_prediction = colorify(rev_prediction, rev_prediction)
        rev_wer = "NA"

    return [rev_res, colored_rev_prediction, rev_wer, time_taken,
            gr.Button(value="Submit", variant="primary", interactive=False)]


def file_transcribe_whisper(audio_file, actual_tr, task, language):
    whisper_res, time_taken = get_transcription_whisper(audio_file, task, get_lang_code(language))
    whisper_prediction = whisper_res.text

    if (actual_tr != ""):
        colored_whisper_prediction = colorify(actual_tr, whisper_prediction)
        whisper_wer = get_wer(whisper_prediction, actual_tr)
    else:
        colored_whisper_prediction = colorify(whisper_prediction, whisper_prediction)
        whisper_wer = "NA"
    

    if (task == "transcribe"):
        return [whisper_res.text, whisper_res.words, colored_whisper_prediction, whisper_wer, time_taken,
                gr.Button(value="Submit", variant="primary", interactive=False)]
    else:
        return [whisper_res.text, None, colored_whisper_prediction, whisper_wer, time_taken,
                gr.Button(value="Submit", variant="primary", interactive=False)]


def file_transcribe_aws(audio_file, actual_tr, task, language):
    if (task == "translate"):
        return ["", "{TRANSLATION NOT SUPPORTED NATIVELY}", "NA", 0.00,
                gr.Button(value="Submit", variant="primary", interactive=False)]
    else:
        # lang code handling
        if (get_lang_code(language) == "en"):
            language = "en-US"
        elif (get_lang_code(language) == "hi"):
            language = "hi-IN"
        elif (get_lang_code(language) == "es"):
            language = "es-ES"
        elif (get_lang_code(language) == None):
            language = None

        aws_res, time_taken = get_transcription_aws(audio_file, language)
        aws_prediction = aws_res["results"]["transcripts"][0]["transcript"]

        if (actual_tr != ""):
            colored_aws_prediction = colorify(actual_tr, aws_prediction)
            aws_wer = get_wer(aws_prediction, actual_tr)
        else:
            colored_aws_prediction = colorify(aws_prediction, aws_prediction)
            aws_wer = "NA"
        
        return [aws_res, colored_aws_prediction, aws_wer, time_taken,
                gr.Button(value="Submit", variant="primary", interactive=False)]

    

########################## JS FUNCTIONS ##########################

disable_play_and_clear_buttons = """
function() {
    document.querySelector(".play-pause-button").disabled = true
    document.querySelector(".play-pause-button").addEventListener("mouseover", function() {
        document.querySelector(".play-pause-button").style.color = "#9ca3af";
    })
    document.querySelector(".play-pause-button").style.cursor = "no-drop";

    document.querySelector("[aria-label='Clear']").disabled = true
    document.querySelector("[aria-label='Clear']").style.cursor = "no-drop";
}
"""

enable_play_and_clear_buttons = """
function(simplismart_res, assembly_res, deepgram_res, rev_res, whisper_res_words, aws_res) {
    console.log(simplismart_res, assembly_res, deepgram_res, rev_res, whisper_res_words, aws_res)
    if (simplismart_res != undefined &&
        assembly_res != undefined &&
        deepgram_res != undefined &&
        rev_res != undefined &&
        whisper_res_words != undefined &&
        aws_res != undefined) {

        document.querySelector(".play-pause-button").disabled = false
        document.querySelector(".play-pause-button").addEventListener("mouseover", function() {
            document.querySelector(".play-pause-button").style.color = "#2563eb";
        })
        document.querySelector(".play-pause-button").addEventListener("mouseout", function() {
            document.querySelector(".play-pause-button").style.color = "#9ca3af";
        })
        document.querySelector(".play-pause-button").style.cursor = "default";

        document.querySelector("[aria-label='Clear']").disabled = false
        document.querySelector("[aria-label='Clear']").style.cursor = "default";
    }
}
"""

get_current_timestamp_and_scroll = """
function(current_time, res, tr, task, state, text=null) {
    if (res !== undefined && res !== null && task != "translate") {
        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: 'smooth'
        });
    }
    let current_time_arr = document.getElementById("time").innerText.split(":")
    let current_time_in_s = (60 * parseInt(current_time_arr[0])) + parseInt(current_time_arr[1]);
    return [current_time_in_s, res, tr, task, state, text];
}
"""

########################## TRANSCRIPTION HANDLING ##########################

def get_actual_transcription(audio_file):

    filename = audio_file.split("/")[-1]

    transcriptions = {
        "simple_english.wav": {
            "transcription": "Surprising and powerful ways for him to help you or your company. Okay, enough fluffy metaphors, let's clarify some terms. AI, as you probably know, stands for Artificial Intelligence.",
            "task": "transcribe",
            "language": "English"
        },
        "numbers_detection_1.wav": {
            "transcription": "Bob has 100 shares of Ed's Carpets, and thinks they are fairly priced at $20, or maybe even do for a fall. Bob agrees to enter a contract to sell his shares to Ken for $22 in a month. The fee that Ken pays Bob for the option, called the premium, is $200 or the $2 per share difference multiplied by 100 shares. If the price of Ed's Carpets shoots up to $30 in a month, Ken can exercise the option.",
            "task": "transcribe",
            "language": "English"
        },
        "numbers_detection_2.wav": {
            "transcription": "EPS, $2.18 that is a beat. The street was close to $2.11. And importantly, revenue is up 2% to a 119.6 billion. That is also a beat and does break that trend. So Apple returning to growth here on the top line. Also critically, gross margin's 45.9%, that is also better than expected. Let's just turn quickly to the segments. iPhone clocks in at 69.7 billion, that is better than consensus.",
            "task": "transcribe",
            "language": "English"
        },
        "multiple_speakers.wav": {
            "transcription": "Bella, Gloria, love. Oh. How are you? Oh, I'm okay. I will be. I said she could stay with us tomorrow, just until she feels better. Yeah, of course she can. No, things won't be for long. Well, you can stay for as long as you want, my love. I've really missed you. Pops. Great to see you, love.",
            "task": "transcribe",
            "language": "English"
        },
        "hindi_transcription.wav": {
            "transcription": "चिन्ता को तलवार की नोक पे रखे वो राजपूत। जो अंगारों पे चले फिर भी मूचों को ताओ दे, वो राजपूत। रेत की नाओ लेकर समंदर से शर्त लगाए वो राजपूत। और जिसका सर कटे, फिर भी धड दुश्मन से लड़ता रहे वो, वो राजपूत।",
            "task": "transcribe",
            "language": "Hindi"
        },
        "hindi_translation.wav": {
            "transcription": "Whenever the cloud of pain loomed, when the shadow of sorrow waved, when tears reached the eyelids, when this lonely heart got scared, we explained to the heart, why does the heart cry? Only you happen in the world.",
            "task": "translate",
            "language": "Hindi"
        },
        "company_names.wav": {
            "transcription": "You're using a browser right now to watch this video, and some of the popular ones are Firefox, Google Chrome, and Safari, but in 1992, Erwise was created. Erwise was an Internet browser and the first to have a graphical interface. A few browsers came before and after but, in 1993, Mosaic was created and it would popularise surfing the web. Mosaic influenced many of the browsers to follow including Netscape Navigator in 1994. This became the most popular web browser at the time, accounting for 90 percent of the web usage in 1995. In the early 90s, companies like AOL and CompuServe were starting to provide dial-up Internet access.",
            "task": "transcribe",
            "language": "English"
        },
        "spanish_audio.wav": {
            "transcription": "Ehh mamá una cosa, voy a ir al supermercado por si quieres que compre algo, había pensado en comprar dos barras de pan para el desayuno, unos tomates, unas cebollas y lo que no sé es si quedan patatas. Si lo puedes mirar y me avisas te lo agradezco. Luego también si necesitas alguna otra cosa me lo escribes en un mensaje o me llamas pero rápido porque voy a ir ya. Chao, te quiero.",
            "task": "transcribe",
            "language": "Spanish"
        }
    }

    if filename in transcriptions:
        return [transcriptions[filename]["transcription"], 
                transcriptions[filename]["task"], 
                transcriptions[filename]["language"]]
    else:
        return ["", "transcribe", "English"]

########################## WORD-BY-WORD ##########################

def get_words_simplismart(current_time, response, transcription, task, state):

    if (task == "transcribe"):
        state = []

        if (response != None):
            # get the initial phrase
            i = 0
            while (response["words"][i]["start"] <= float(current_time)+response["words"][0]["start"]):
                state.append(response["words"][i]["word"])
                i += 1

            # continuously append to initial phrase
            while (i < len(response["words"])):
                state.append(response["words"][i]["word"])
                time.sleep(abs(response["words"][i]["end"] - response["words"][i]["start"]))
                if (i+1 != len(response["words"])):
                    gap_time = abs(response["words"][i+1]["start"] - response["words"][i]["end"])
                    time.sleep(gap_time - 0.2*(gap_time))
                yield current_time, response, (" ".join(state)), task, state
                i += 1
        else:
            yield current_time, response, transcription, task, state
    else:
        yield current_time, response, transcription, task, state

def word_by_word_simplismart(current_time, response, transcription, task, state):
    if (task == "transcribe"):
        result = []

        if (response != None):
            # get the initial phrase
            i = 0
            while (response["words"][i]["start"] <= float(current_time)+response["words"][0]["start"]):
                result.append((response["words"][i]["word"], None))
                i += 1

            # continuously append to initial phrase
            while (i < len(response["words"])):
                result.append((response["words"][i]["word"], None))
                time.sleep(abs(response["words"][i]["end"] - response["words"][i]["start"]))
                if (i+1 != len(response["words"])):
                    gap_time = abs(response["words"][i+1]["start"] - response["words"][i]["end"])
                    time.sleep(gap_time - 0.2*(gap_time))
                yield current_time, response, result, task, state
                i += 1
        else:
            yield current_time, response, transcription, task, state
    else:
        yield current_time, response, transcription, task, state


def get_words_deepgram(current_time, response, transcription, task, state):
    
    if (task == "transcribe"):
        state = []

        if (response != None):
            word_chunks = response["results"]["channels"][0]["alternatives"][0]["words"]
            words = response["results"]["channels"][0]["alternatives"][0]["transcript"].split(" ")

            # get the initial phrase
            i = 0
            while (word_chunks[i]["start"] < float(current_time)+word_chunks[0]["start"]):
                state.append(words[i])
                i += 1

            # continuously append to initial phrase
            while (i < len(word_chunks)):
                state.append(words[i])
                time.sleep(word_chunks[i]["end"] - word_chunks[i]["start"])
                if (i+1 != len(word_chunks)):
                    gap_time = word_chunks[i+1]["start"] - word_chunks[i]["end"]
                    time.sleep(gap_time - 0.2*(gap_time))
                yield current_time, response, (" ".join(state)), task, state
                i += 1
        else:
            yield current_time, response, transcription, task, state
    else:
        yield current_time, response, transcription, task, state


def get_words_assembly(current_time, response, transcription, task, state):

    if (task == "transcribe"):
        state = []

        if (response != None):
            word_chunks = response

            # get the initial phrase
            i = 0
            while (word_chunks[i]["start"] < float(current_time)+word_chunks[0]["start"]/1000):
                state.append(word_chunks[i]["text"])
                i += 1

            # continuously append to initial phrase
            while (i < len(word_chunks)):
                state.append(word_chunks[i]["text"])
                time.sleep(word_chunks[i]["end"] - word_chunks[i]["start"])
                if (i+1 != len(word_chunks)):
                    gap_time = word_chunks[i+1]["start"] - word_chunks[i]["end"]
                    time.sleep(gap_time - 0.2*(gap_time))
                yield current_time, response, (" ".join(state)), task, state
                i += 1
        else:
            yield current_time, response, transcription, task, state
    else:
        yield current_time, response, transcription, task, state
    


def get_words_rev_ai(current_time, response, transcription, task, state):

    if (task == "transcribe"):
        state = []

        if (response != None):
            word_chunks = response["monologues"][0]["elements"]

            # get the initial phrase
            i = 0
            while True:
                if (word_chunks[i]["type"] == "text"):
                    if (word_chunks[i]["ts"] < float(current_time)+word_chunks[0]["ts"]):
                        state.append(word_chunks[i]["value"])
                    else:
                        break
                elif (word_chunks[i]["type"] == "punct"):
                    state.append(word_chunks[i]["value"])
                i += 1

            # continuously append to initial phrase
            while (i < len(word_chunks)):
                if (word_chunks[i]["type"] == "text"):
                    state.append(word_chunks[i]["value"])
                    time.sleep(word_chunks[i]["end_ts"] - word_chunks[i]["ts"])

                    j = i
                    while (i+1 != len(word_chunks) and word_chunks[i+1]["type"] == "punct"):
                        state.append(word_chunks[i+1]["value"])
                        i += 1
                    
                    if (i+1 != len(word_chunks)):
                        gap_time = word_chunks[i+1]["ts"] - word_chunks[j]["end_ts"]
                        time.sleep(gap_time - 0.2*(gap_time))

                yield current_time, response, ("".join(state)), task, state
                i += 1
        
        else:
            yield current_time, response, transcription, task, state
    else:
        yield current_time, response, transcription, task, state


def get_words_whisper(current_time, response_words, transcription, task, state, response_text):

    if (task == "transcribe"):
        state = []

        if (response_words != None):
            word_chunks = response_words
            words = response_text.split(" ")

            # get the initial phrase
            i = 0
            while (word_chunks[i]["start"] < float(current_time)+word_chunks[0]["start"]):
                state.append(words[i])
                i += 1

            # continuously append to initial phrase
            while (i < min(len(word_chunks), len(words))):
                state.append(words[i])
                time.sleep(word_chunks[i]["end"] - word_chunks[i]["start"])
                if (i+1 != len(word_chunks)):
                    gap_time = word_chunks[i+1]["start"] - word_chunks[i]["end"]
                    time.sleep(gap_time - 0.2*(gap_time))
                yield current_time, response_words, (" ".join(state)), task, state, response_text
                i += 1
        else:
            yield current_time, response_words, transcription, task, state, response_text
    else:
        yield current_time, response_words, transcription, task, state, response_text


def get_words_aws(current_time, response, transcription, task, state):
    if (task == "transcribe"):
        state = []

        if (response != None):
            word_chunks = response["results"]["items"]
            
            # get the initial phrase
            i = 0
            while True:
                if (word_chunks[i]["type"] == "pronunciation"):
                    if (float(word_chunks[i]["start_time"]) < float(current_time)+float(word_chunks[0]["start_time"])):
                        state.append(" " + word_chunks[i]["alternatives"][0]["content"])
                    else:
                        break
                elif (word_chunks[i]["type"] == "punctuation"):
                    state.append(word_chunks[i]["alternatives"][0]["content"])
                i += 1

            # continuously append to initial phrase
            while (i < len(word_chunks)):
                if (word_chunks[i]["type"] == "pronunciation"):
                    state.append(" " + word_chunks[i]["alternatives"][0]["content"])
                    time.sleep(float(word_chunks[i]["end_time"]) - float(word_chunks[i]["start_time"]))

                    j = i
                    while (i+1 != len(word_chunks) and word_chunks[i+1]["type"] == "punctuation"):
                        state.append(word_chunks[i+1]["alternatives"][0]["content"])
                        i += 1
                    
                    if (i+1 != len(word_chunks)):
                        gap_time = float(word_chunks[i+1]["start_time"]) - float(word_chunks[j]["end_time"])
                        time.sleep(gap_time - 0.2*(gap_time))

                yield current_time, response, ("".join(state)), task, state
                i += 1
        else:
            yield current_time, response, transcription, task, state
    else:
        yield current_time, response, transcription, task, state


########################## CLEAR FUNCTION ##########################

def clear_all(*outputs):
    return [None]*len(outputs)

def enable_submit_button():
    return gr.Button(value="Submit", variant="primary", interactive=True)

def disable_submit_button():
    return gr.Button(value="Submit", variant="primary", interactive=False)

########################## AUDIO STOP FUNCTION ##########################

def get_colorified_transcript(actual_tr, simplismart_res, assembly_res, deepgram_res, rev_res, whisper_res_text, aws_res):
    # print(simplismart_res, "\n")
    # print(assembly_res, "\n")
    # print(deepgram_res, "\n")
    # print(rev_res, "\n")
    # print(whisper_res_text, "\n")
    # print(aws_res, "\n")
    # print(type(simplismart_res), type(assembly_res), type(deepgram_res), type(rev_res), type(whisper_res_text), type(aws_res))
    
    simplismart_prediction = ""
    for item in simplismart_res["transcription"]:
        simplismart_prediction += item["text"]
    
    assembly_prediction = ""
    for item in assembly_res:
        assembly_prediction += item["text"] + " "

    deepgram_prediction = deepgram_res["results"]["channels"][0]["alternatives"][0]["transcript"]

    rev_prediction = ""
    for item in rev_res["monologues"][0]["elements"]:
        rev_prediction += item["value"]

    whisper_prediction = whisper_res_text
    aws_prediction = aws_res["results"]["transcripts"][0]["transcript"]

    if (actual_tr != ""):
        colorified_simplismart_transcript = colorify(actual_tr, simplismart_prediction)
        colorified_deepgram_transcript = colorify(actual_tr, deepgram_prediction)
        colorified_assembly_transcript = colorify(actual_tr, assembly_prediction)
        colorified_rev_transcript = colorify(actual_tr, rev_prediction)
        colorified_whisper_transcript = colorify(actual_tr, whisper_prediction)
        colorified_aws_transcript = colorify(actual_tr, aws_prediction)
    
    else:
        colorified_simplismart_transcript = colorify(simplismart_prediction, simplismart_prediction)
        colorified_deepgram_transcript = colorify(deepgram_prediction, deepgram_prediction)
        colorified_assembly_transcript = colorify(assembly_prediction, assembly_prediction)
        colorified_rev_transcript = colorify(rev_prediction, rev_prediction)
        colorified_whisper_transcript = colorify(whisper_prediction, whisper_prediction)
        colorified_aws_transcript = colorify(aws_prediction, aws_prediction)
    
    return [colorified_simplismart_transcript, colorified_deepgram_transcript, colorified_assembly_transcript,
            colorified_rev_transcript, colorified_whisper_transcript, colorified_aws_transcript]
    

########################## UI ##########################

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.Color(
        name="simplismart-blue",
        c50="#eff6ff",
        c100="#dbeafe",
        c200="#bfdbfe",
        c300="#93c5fd",
        c400="#60a5fa",
        c500="#1e90ff",
        c600="#2563eb",
        c700="#2056ed",
        c800="#1e40af",
        c900="#1e3a8a",
        c950="#1d3660",
    ),
    secondary_hue=gr.themes.colors.Color(
        name="simplismart-blue",
        c50="#eff6ff",
        c100="#dbeafe",
        c200="#bfdbfe",
        c300="#93c5fd",
        c400="#60a5fa",
        c500="#1e90ff",
        c600="#2563eb",
        c700="#2056ed",
        c800="#1e40af",
        c900="#1e3a8a",
        c950="#1d3660",
    ),
    neutral_hue=gr.themes.colors.Color(
        name="simplismart-dark",
        c50="#f8fafc",
        c100="#f1f5f9",
        c200="#e2e8f0",
        c300="#cbd5e1",
        c400="#94a3b8",
        c500="#64748b",
        c600="#1F1C30",
        c700="#1D1B2E",
        c800="#141321",
        c900="#100F1A",
        c950="#050509",
    ),
    font=gr.themes.GoogleFont(name="Inter",
                              weights=(200,800))
)

# neutral_hue=gr.themes.colors.Color(
#         name="simplismart-dark",
#         c50="#f8fafc",
#         c100="#f1f5f9",
#         c200="#e2e8f0",
#         c300="#cbd5e1",
#         c400="#94a3b8",
#         c500="#64748b",
#         c600="#1c234b",
#         c700="#181b3c",
#         c800="#141530",
#         c900="#101123",
#         c950="#0e0e19",
#     )

css = """
@font-face {
    font-family: "Poppins";
    src: url("/file=Poppins-SemiBold.ttf");
}

@font-face {
    font-family: "Cal Sans";
    src: url("/file=calsans-semibold.otf") format("opentype");
}

#heading h1 {
    font-family: "Cal Sans" !important;
    font-size: 40px !important;
    margin: 0px !important;
}

#heading h2 {
    font-family: "Cal Sans" !important;
    font-size: 40px !important;
    margin: 0px !important;
    color:#1D4ED8 !important;
}

#logo {
    border:none !important;
    background:black !important;
    margin-top: 30px;
    margin-bottom: 30px;
}

#logo .image-container {
    display:flex;
    justify-content:center;
}

#logo img {
    width: 200px;
}

.paginate {
    justify-content:flex-start !important;
}

.paginate > button {
    padding:3px;
    border:1px solid #374151;
    border-radius:3px;
}

.paginate .current-page {
    color:#f97316;
}

.rewind {
    display: None;
}

.skip {
    display: None;
}

.fix-wer-time-col {
    min-width: min(200px,100%) !important;
}

.fix-wer-time-tb > label {
    display:flex !important;
    align-items:center !important;
}

.fix-wer-time-tb > label > span {
    min-width:70px !important;
    margin-bottom:0px !important;
}

.fix-rtb {
    min-height:125px !important;
}

.fix-rtb span[style*="color:red"] {
    color:white !important;
    background-color:#a60505;
    padding:2px;
    border-radius:2px;
}

.fix-rtb span[style*="color:green"] {
    color:white !important;
    background-color:#085e02;
    padding:2px;
    border-radius:2px;
}

.fix-rtb div[role="textbox"] {
  line-height: 1.7;
  min-height:75px;
}

.fix-htb .textfield {
    word-break: keep-all;
}

.fix-htb span {
    color: white;
}

span[data-testid="block-info"] {
  color: #017FF9;
  font-weight:500;
  font-family:"Cal Sans" !important;
}

"""

disable_audio_input_buttons = """
function () {
    if (document.getElementById("audio-file-input") != null) {
        document.querySelector(".playback").disabled = true;
        document.querySelectorAll(".action").forEach(function(button) {
            button.disabled = true;
        })
    }
}
"""


def main():
    submit_button = gr.Button(value="Submit", variant="primary", interactive=False, elem_id=["submit-button"])
    task = gr.Dropdown(label="Task", choices=["transcribe", "translate"], value="transcribe", interactive=True)
    language = gr.Dropdown(label="Input Language", choices=["Spanish", "English", "Hindi"], value="English", interactive=True)

    with gr.Blocks(css=css, theme=theme) as demo:
        gr.Markdown(
            """
            ## Speech-To-Text 
            # Quality Benchmarking
            """,
            elem_id="heading"
        )

        with gr.Row(variant="default"):

            with gr.Column(scale=2, variant="panel"):
                file_input = gr.Audio(sources=["upload"], type="filepath", label="Audio File", elem_id="audio-file-input")
                actual_transcription = gr.Textbox(label="Golden Transcription (Optional)", max_lines=5, autoscroll=False)

                examples = gr.Examples(
                    label="Example Audios (.mp3 files)",
                    examples=[
                        ["audio_files/simple_english.mp3"],
                        ["audio_files/numbers_detection_1.mp3"],
                        ["audio_files/numbers_detection_2.mp3"],
                        ["audio_files/multiple_speakers.mp3"],
                        ["audio_files/hindi_transcription.mp3"],
                        ["audio_files/hindi_translation.mp3"],
                        ["audio_files/company_names.mp3"],
                        ["audio_files/spanish_audio.mp3"]
                    ],
                    inputs=[file_input],
                    examples_per_page=8,
                    example_labels=["simple_english", 
                                    "detect_numbers1", 
                                    "detect_numbers2",
                                    "multiple_speakers",
                                    "hindi_transcription",
                                    "hindi_translation",
                                    "company_names",
                                    "spanish_audio"],
                    elem_id="example-audio-files"
                )

                submit_button.render()

                with gr.Accordion("Simplismart Parameter Configuration", open=False):
                    with gr.Accordion("Base Parameters", open=True) as base_parameters:
                        task.render()
                        language.render()
                        initial_prompt = gr.Textbox(label="Initial Prompt", placeholder="Enter prompt", interactive=True)
                        beam_size = gr.Number(label="Beam Size", value=5, interactive=True)
                        best_of = gr.Number(label="Best of", value=5, interactive=True)
                        patience = gr.Slider(label="Patience", minimum=0.0, maximum=1.0, value=1, interactive=True)
                    with gr.Accordion("VAD Parameters", open=False) as vad_parameters:
                        vad_onset = gr.Slider(label="VAD Onset", minimum=0.000, maximum=1.000, value=0.500, interactive=True)
                        vad_offset = gr.Slider(label="VAD Offset", minimum=0.000, maximum=1.000, value=0.363, interactive=True)
                        min_duration_off = gr.Number(label="Min. Silence Duration (s)", value=0, minimum=0, interactive=True)
                        min_duration_on = gr.Number(label="Min. Speech Duration (s)", value=0, minimum=0, interactive=True)
                        pad_onset = gr.Number(label="Pad Onset (s)", value=0, minimum=0, interactive=True)
                        pad_offset = gr.Number(label="Pad Offset (s)", value=0, minimum=0, interactive=True)
                

            with gr.Column(scale=5, variant="panel"):
                current_time = gr.Textbox(value="0", visible=False, show_label=False, elem_id="current-time-tracker", 
                                          interactive=False, container=False)

                with gr.Row(variant=["compact"]):
                    with gr.Column(scale=5):
                        simplismart_res = gr.JSON(visible=False)
                        simplismart_tr = RichTextbox(label="Simplismart", interactive=False, elem_classes=["fix-rtb"])
                        # simplismart_tr = gr.HighlightedText(label="Simplismart Transcription", 
                        #                                     combine_adjacent=True, 
                        #                                     adjacent_separator=" ", 
                        #                                     show_legend=False, 
                        #                                     color_map={"+": "#34821d", "-": "#b50d1e"}, 
                        #                                     scale=5,
                        #                                     interactive=False,
                        #                                     elem_classes=["fix-htb"])
                    with gr.Column(scale=1, elem_classes=["fix-wer-time-col"]):
                        simplismart_wer = gr.Textbox(label="WER (%)", min_width=100, elem_classes=["fix-wer-time-tb"])
                        simplismart_tt = gr.Textbox(label="Time (s)", min_width=100, elem_classes=["fix-wer-time-tb"])
                        simplismart_state = gr.State([])
                with gr.Row():
                    with gr.Column(scale=5):
                        deepgram_res = gr.JSON(visible=False)
                        deepgram_tr = RichTextbox(label="Deepgram", interactive=False, elem_classes=["fix-rtb"])
                    with gr.Column(scale=1, elem_classes=["fix-wer-time-col"]):
                        deepgram_wer = gr.Textbox(label="WER (%)", interactive=False, min_width=100, elem_classes=["fix-wer-time-tb"])
                        deepgram_tt = gr.Textbox(label="Time (s)", interactive=False, min_width=100, elem_classes=["fix-wer-time-tb"])
                        deepgram_state = gr.State([])
                with gr.Row():
                    with gr.Column(scale=5):
                        assembly_res = gr.JSON(visible=False)
                        assembly_tr = RichTextbox(label="Assembly AI", interactive=False, elem_classes=["fix-rtb"])
                    with gr.Column(scale=1, elem_classes=["fix-wer-time-col"]):
                        assembly_wer = gr.Textbox(label="WER (%)", interactive=False, min_width=100, elem_classes=["fix-wer-time-tb"])
                        assembly_tt = gr.Textbox(label="Time (s)", interactive=False, min_width=100, elem_classes=["fix-wer-time-tb"])
                        assembly_state = gr.State([])
                with gr.Row():
                    with gr.Column(scale=5):
                        rev_res = gr.JSON(visible=False)
                        rev_tr = RichTextbox(label="Rev AI", interactive=False, elem_classes=["fix-rtb"])
                    with gr.Column(scale=1, elem_classes=["fix-wer-time-col"]):
                        rev_wer = gr.Textbox(label="WER (%)", interactive=False, min_width=100, elem_classes=["fix-wer-time-tb"])
                        rev_tt = gr.Textbox(label="Time (s)", interactive=False, min_width=100, elem_classes=["fix-wer-time-tb"])
                        rev_state = gr.State([])
                with gr.Row():
                    with gr.Column(scale=5):
                        whisper_res_text = gr.Textbox(visible=False)
                        whisper_res_words = gr.JSON(visible=False)
                        whisper_tr = RichTextbox(label="Open AI Whisper", scale=5, interactive=False, elem_classes=["fix-rtb"])
                    with gr.Column(scale=1, elem_classes=["fix-wer-time-col"]):
                        whisper_wer = gr.Textbox(label="WER (%)", interactive=False, min_width=100, elem_classes=["fix-wer-time-tb"])
                        whisper_tt = gr.Textbox(label="Time (s)", interactive=False, min_width=100, elem_classes=["fix-wer-time-tb"])
                        whisper_state = gr.State([])
                with gr.Row():
                    with gr.Column(scale=5):
                        aws_res = gr.JSON(visible=False)
                        aws_tr = RichTextbox(label="AWS Transcribe", scale=5, interactive=False, elem_classes=["fix-rtb"])
                    with gr.Column(scale=1, elem_classes=["fix-wer-time-col"]):
                        aws_wer = gr.Textbox(label="WER (%)", interactive=False, min_width=100, elem_classes=["fix-wer-time-tb"])
                        aws_tt = gr.Textbox(label="Time (s)", interactive=False, min_width=100, elem_classes=["fix-wer-time-tb"])
                        aws_state = gr.State([])

        gr.Image(value="css/Logo.svg", type='filepath', interactive=False, show_label=False, 
                 show_download_button=False, elem_id="logo")
            
        # EVENT LISTENER FUNCTIONALITY

        # disable the submit button -> clear everything -> get new audio + tr -> enable submit button

        gr.on(
            triggers=[examples.dataset.click, file_input.upload],
            fn=disable_submit_button,
            inputs=None,
            outputs=[submit_button]
        ).then(
            fn=clear_all,
            inputs=[simplismart_res, simplismart_tr, simplismart_wer, simplismart_tt,
                    deepgram_res, deepgram_tr, deepgram_wer, deepgram_tt, 
                    assembly_res, assembly_tr, assembly_wer, assembly_tt,
                    rev_res, rev_tr, rev_wer, rev_tt, 
                    whisper_res_text, whisper_res_words, whisper_tr, whisper_wer, whisper_tt, 
                    aws_res, aws_tr, aws_wer, aws_tt],
            outputs=[simplismart_res, simplismart_tr, simplismart_wer, simplismart_tt,
                    deepgram_res, deepgram_tr, deepgram_wer, deepgram_tt, 
                    assembly_res, assembly_tr, assembly_wer, assembly_tt,
                    rev_res, rev_tr, rev_wer, rev_tt, 
                    whisper_res_text, whisper_res_words, whisper_tr, whisper_wer, whisper_tt, 
                    aws_res, aws_tr, aws_wer, aws_tt]
        ).then(
            fn=get_actual_transcription,
            inputs=[file_input],
            outputs=[actual_transcription, task, language]
        ).then(
            fn=enable_submit_button,
            inputs=None,
            outputs=[submit_button]
        ).then(
            fn=None,
            inputs=None,
            outputs=None,
            js=disable_audio_input_buttons
        )

        # examples.dataset.click(
        #     fn=disable_submit_button,
        #     inputs=None,
        #     outputs=[submit_button]
        # ).then(
        #     fn=clear_all,
        #     inputs=[simplismart_res, simplismart_tr, simplismart_wer, simplismart_tt,
        #             deepgram_res, deepgram_tr, deepgram_wer, deepgram_tt, 
        #             assembly_res, assembly_tr, assembly_wer, assembly_tt,
        #             rev_res, rev_tr, rev_wer, rev_tt, 
        #             whisper_res_text, whisper_res_words, whisper_tr, whisper_wer, whisper_tt, 
        #             aws_res, aws_tr, aws_wer, aws_tt],
        #     outputs=[simplismart_res, simplismart_tr, simplismart_wer, simplismart_tt,
        #             deepgram_res, deepgram_tr, deepgram_wer, deepgram_tt, 
        #             assembly_res, assembly_tr, assembly_wer, assembly_tt,
        #             rev_res, rev_tr, rev_wer, rev_tt, 
        #             whisper_res_text, whisper_res_words, whisper_tr, whisper_wer, whisper_tt, 
        #             aws_res, aws_tr, aws_wer, aws_tt]
        # ).then(
        #     fn=get_actual_transcription,
        #     inputs=[file_input],
        #     outputs=[actual_transcription, task, language]
        # ).then(
        #     fn=enable_submit_button,
        #     inputs=None,
        #     outputs=[submit_button]
        # )

        # file_input.change(
        #     fn=None,
        #     inputs=None,
        #     outputs=None,
        #     js=disable_audio_input_buttons
        # ).then(
        #     fn=enable_submit_button,
        #     inputs=None,
        #     outputs=[submit_button]
        # )




        # file submission + play events
        # disable play button -> get API output -> enable play button once all are updated.

        submit_button.click(
            fn=None,
            inputs=None,
            outputs=None,
            js=disable_play_and_clear_buttons
        )
        
        # simplismart
        submit_button.click(
            fn=file_transcribe_simplismart,
            inputs=[file_input, actual_transcription, task, language, initial_prompt, beam_size, best_of, patience, vad_onset, vad_offset, min_duration_on, min_duration_off, pad_onset, pad_offset],
            outputs=[simplismart_res, simplismart_tr, simplismart_wer, simplismart_tt, submit_button],
            scroll_to_output=True,
            concurrency_limit='default'
        ).then(
            fn=None,
            inputs=[simplismart_res, assembly_res, deepgram_res, rev_res, whisper_res_words, aws_res],
            outputs=None,
            js=enable_play_and_clear_buttons
        )

        # deepgram
        submit_button.click(
            fn=file_transcribe_deepgram,
            inputs=[file_input, actual_transcription, task, language],
            outputs=[deepgram_res, deepgram_tr, deepgram_wer, deepgram_tt, submit_button],
            scroll_to_output=True,
            concurrency_limit='default'
        ).then(
            fn=None,
            inputs=[simplismart_res, assembly_res, deepgram_res, rev_res, whisper_res_words, aws_res],
            outputs=None,
            js=enable_play_and_clear_buttons
        )

        # assembly ai
        submit_button.click(
            fn=file_transcribe_assemblyai,
            inputs=[file_input, actual_transcription, task, language],
            outputs=[assembly_res, assembly_tr, assembly_wer, assembly_tt, submit_button],
            scroll_to_output=True,
            concurrency_limit='default'
        ).then(
            fn=None,
            inputs=[simplismart_res, assembly_res, deepgram_res, rev_res, whisper_res_words, aws_res],
            outputs=None,
            js=enable_play_and_clear_buttons
        )

        # rev ai
        submit_button.click(
            fn=file_transcribe_rev_ai,
            inputs=[file_input, actual_transcription, task, language],
            outputs=[rev_res, rev_tr, rev_wer, rev_tt, submit_button],
            scroll_to_output=True,
            concurrency_limit='default'
        ).then(
            fn=None,
            inputs=[simplismart_res, assembly_res, deepgram_res, rev_res, whisper_res_words, aws_res],
            outputs=None,
            js=enable_play_and_clear_buttons
        )

        # whisper
        submit_button.click(
            fn=file_transcribe_whisper,
            inputs=[file_input, actual_transcription, task, language],
            outputs=[whisper_res_text, whisper_res_words, whisper_tr, whisper_wer, whisper_tt, submit_button],
            scroll_to_output=True,
            concurrency_limit='default'
        ).then(
            fn=None,
            inputs=[simplismart_res, assembly_res, deepgram_res, rev_res, whisper_res_words, aws_res],
            outputs=None,
            js=enable_play_and_clear_buttons
        )

        # aws
        submit_button.click(
            fn=file_transcribe_aws,
            inputs=[file_input, actual_transcription, task, language],
            outputs=[aws_res, aws_tr, aws_wer, aws_tt, submit_button],
            scroll_to_output=True,
            concurrency_limit='default'
        ).then(
            fn=None,
            inputs=[simplismart_res, assembly_res, deepgram_res, rev_res, whisper_res_words, aws_res],
            outputs=None,
            js=enable_play_and_clear_buttons
        )


        # word-by-word

        play_event_simplismart = file_input.play(
            fn=get_words_simplismart,
            inputs=[current_time, simplismart_res, simplismart_tr, task, simplismart_state],
            outputs=[current_time, simplismart_res, simplismart_tr, task, simplismart_state],
            js=get_current_timestamp_and_scroll,
            concurrency_limit='default'
        )
        play_event_deepgram = file_input.play(
            fn=get_words_deepgram,
            inputs=[current_time, deepgram_res, deepgram_tr, task, deepgram_state],
            outputs=[current_time, deepgram_res, deepgram_tr, task, deepgram_state],
            js=get_current_timestamp_and_scroll,
            concurrency_limit='default'
        )
        play_event_assembly = file_input.play(
            fn=get_words_assembly,
            inputs=[current_time, assembly_res, assembly_tr, task, assembly_state],
            outputs=[current_time, assembly_res, assembly_tr, task, assembly_state],
            js=get_current_timestamp_and_scroll,
            concurrency_limit='default'
        )
        play_event_rev = file_input.play(
            fn=get_words_rev_ai,
            inputs=[current_time, rev_res, rev_tr, task, rev_state],
            outputs=[current_time, rev_res, rev_tr, task, rev_state],
            js=get_current_timestamp_and_scroll,
            concurrency_limit='default'
        )
        play_event_whisper = file_input.play(
            fn=get_words_whisper,
            inputs=[current_time, whisper_res_words, whisper_tr, task, whisper_state, whisper_res_text],
            outputs=[current_time, whisper_res_words, whisper_tr, task, whisper_state, whisper_res_text],
            js=get_current_timestamp_and_scroll,
            concurrency_limit='default'
        )
        play_event_aws = file_input.play(
            fn=get_words_aws,
            inputs=[current_time, aws_res, aws_tr, task, aws_state],
            outputs=[current_time, aws_res, aws_tr, task, aws_state],
            js=get_current_timestamp_and_scroll,
            concurrency_limit='default'
        )


        # cancel event
        file_input.pause(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[play_event_simplismart, play_event_deepgram, play_event_assembly, 
                     play_event_rev, play_event_whisper, play_event_aws]
        )

        # file clear
        file_input.clear(
            fn=clear_all,
            inputs=[actual_transcription, 
                    simplismart_res, simplismart_tr, simplismart_wer, simplismart_tt,
                    deepgram_res, deepgram_tr, deepgram_wer, deepgram_tt, 
                    assembly_res, assembly_tr, assembly_wer, assembly_tt,
                    rev_res, rev_tr, rev_wer, rev_tt, 
                    whisper_res_text, whisper_res_words, whisper_tr, whisper_wer, whisper_tt, 
                    aws_res, aws_tr, aws_wer, aws_tt],
            outputs=[actual_transcription, 
                    simplismart_res, simplismart_tr, simplismart_wer, simplismart_tt,
                    deepgram_res, deepgram_tr, deepgram_wer, deepgram_tt, 
                    assembly_res, assembly_tr, assembly_wer, assembly_tt,
                    rev_res, rev_tr, rev_wer, rev_tt, 
                    whisper_res_text, whisper_res_words, whisper_tr, whisper_wer, whisper_tt, 
                    aws_res, aws_tr, aws_wer, aws_tt],
            concurrency_limit='default'
        )

        # get colorified transcript when audio stops
        file_input.stop(
            fn=get_colorified_transcript,
            inputs=[actual_transcription, simplismart_res, assembly_res, deepgram_res, rev_res, whisper_res_text, aws_res],
            outputs=[simplismart_tr, assembly_tr, deepgram_tr, rev_tr, whisper_tr, aws_tr]
        )

    demo.queue(default_concurrency_limit=40).launch(allowed_paths=["css/Poppins-SemiBold.ttf",
                                                                   "css/calsans-semibold.otf"])

main()
