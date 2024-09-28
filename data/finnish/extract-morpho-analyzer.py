import os

from libvoikko import Voikko
Voikko.setLibrarySearchPath("/opt/homebrew/Cellar/libvoikko/4.3.2/lib")
# # Manually set the library path if necessary
# help(Voikko)


# Initialize Voikko for Finnish
voikko = Voikko(u"fi")
# breakpoint()


def translate_pos(fin_pos):
    pos_translation = {
        "nimisana": "noun",
        "teonsana": "verb",
        "laatusana": "adjective",
        "määritesana": "modifier",
        "seikkasana": "adverb",
        "suhdesana": "preposition",
        "asemosana": "pronoun",
        "lukusana": "numeral",
        "huudahdussana": "interjection",
        "konjunktio": "conjunction",
        "partikkeli": "particle",
        "lyhenne": "abbreviation",
        "ulkoolento": "exessive",  # Example of a grammatical case
        "paikannimi" : "place",
        'nimi':"name",
        "sukunimi":"surname"
        # Add more translations as needed
    }
    if not pos_translation[fin_pos]:
        print(fin_pos)
        breakpoint()
    return pos_translation.get(fin_pos, fin_pos)  # Default to original if not found

def analyze_and_segment(voikko, word):
    analysis = voikko.analyze(word)
    if not analysis:
        return None, None, None

    lemma = analysis[0].get('BASEFORM', word)
    pos = translate_pos(analysis[0].get('CLASS', ''))
    wordbases = analysis[0].get('WORDBASES', '')

    segmentation = wordbases.replace('][', '-').strip('[]').replace('+', '').replace('(', '-').replace(')', '')

    analysis_str = f"{lemma}_{pos}"

    return lemma, analysis_str, segmentation

def process_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            freq, word = parts
            lemma, analysis, segmentation = analyze_and_segment(voikko, word)
            if lemma:
                print(f"{lemma}\t{word}\t{analysis}\t{segmentation}\t{freq}")

file_path = 'data/finnish/finnish-morphochallenge.txt'  # Update your file path
process_file(file_path)

voikko.terminate()
# Specify your file path
file_path = 'data/finnish/finnish-morphochallenge.txt'
process_file(file_path)

# Deinitialize Voikko
voikko.terminate()
