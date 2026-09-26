import re
from unidecode import unidecode
from .number_norm import normalize_numbers
from roman import fromRoman
from .urgu import _replacement, _combined
from .transliterate import transliterate

_whitespace_re = re.compile(r"\s+")
_fullstop_re = re.compile(r"(\,|\.|\!|\?){2,}")
_newline_re = re.compile(r"(\\n){2,}")
_simbols_re = re.compile(r"[^A-Za-z0-9,|\.|?|\-|\'|\s|#|@|$|*|=|+|%|/]")
_apostrophe_re = re.compile(
    r"[\‘|’|ʼ|`|´|ʹ|ʻ|ʽ|ʾ|ʿ|ˈ|ˊ|ʹ|΄|՚|᾽|᾿|‘|‛|′|‵|Ꞌ|ꞌ|＇|‘|'|‘|’|’]")

_punctuations_re = re.compile(r"[!|(|)|:|?]")
_c_symbol_re = re.compile(r"(c.)")
_main_symbol_re = re.compile(r"[#|@|$|*|=|+|%|\-|/]")
_websites_re = re.compile(r"(https://.\w+.\w+|http://.\w+.\w+|\
https://www.\w+.\w+|http://www.\w+.\w+)")
_wtf_re = re.compile(r"(\,\.)")


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def replace_apostrophe(text):
    return re.sub(_apostrophe_re, "'", text)


def collapse_symbols(text):
    return re.sub(_simbols_re, "", text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def collapse_fullstop(text):
    return re.sub(_fullstop_re, text[len(text)-1], text)


def collapse_newline(text):
    return re.sub(_fullstop_re, "", text)


def add_fullstop(text):
    return text if text.strip()[-1] == "." else text + " ."


def replace_c(text):
    if text.group(0)[1] != "h":
        return "k"+text.group(0)[1]
    else:
        return text.group(0)


def replace_main_symbols(text):
    symbol = text.group(0)
    if symbol == "#":
        return " hashtag "
    elif symbol == "@":
        return " elektron kuchukcha "
    elif symbol == "$":
        return " dollar "
    elif symbol == "*":
        return "yulduzcha "
    elif symbol == "=":
        return " teng "
    elif symbol == "+":
        return " qo'shuv "
    elif symbol == "%":
        return " foiz "
    elif symbol == "/":
        return " taqsim "
    elif symbol == "-":
        return "-"


def replace_matematical_q(text):
    symbol = text.group(0)
    if symbol[1:] == "=?":
        return symbol.replace("=?", " teng nechchi")
    else:
        return symbol.replace("=", " teng ")


def replace_website_symbols(text):
    word = text.group(0)
    if "http://" in word and "www" not in word:
        word = word.replace("http://", " echtiitiipii ikki nuqta drop drop ")
        return word.replace(".", " nuqta ")
    elif "https://" in word and "www" not in word:
        word = word.replace(
            "https://", " echtiitiipiiess ikki nuqta drop drop ")
        return word.replace(".", " nuqta ")
    elif "http://" in word and "www" in word:
        word = word.replace("http://", " echtiitiipii ikki nuqta drop drop ")
        word = word.replace("www", " uch dabllyu")
        return word.replace(".", " nuqta ")
    elif "https://" in word and "www" in word:
        word = word.replace(
            "https://", " echtiitiipiiess ikki nuqta drop drop ")
        word = word.replace("www", " uch dabllyu")
        return word.replace(".", " nuqta ")


def replace_matematical_symbols(text):
    symbol = text.group(0)
    if symbol.count("+") == 1:
        return symbol.replace("+", " qo'shuv ")
    elif symbol.count("-") == 1 and symbol.count("=") == 1:
        return symbol.replace("-", " ayirilgan ")
    elif symbol.count("-") == 1 and symbol.count("=") == 0:
        return symbol.replace("-", " tiire ")
    elif symbol.count("*") == 1:
        return symbol.replace("*", " ko'paytirish ")
    elif symbol.count("/") == 1:
        return symbol.replace("/", " bo'lingan ")


def non_uzbek(text):
    return text.replace("w", "dabllyu ")


def i_character(text):
    if text.group(0) in _replacement:
        return _replacement[text.group(0)]
    else:
        return text.group(0)

def replace_s(text):
     return re.sub(r"[?!:]", " .", text)

allowed_characters = "'.,abcdefghijklmnopqrstuvwxyz- "

def remove_disallowed_chars(text):
    text = ''.join([char for char in text if char in allowed_characters])
    return text

vocab_uzb= {
    "a": "a",
    "b": "bee",
    "d": "dee",
    "e": "ee",
    "f": "fee",
    "g": "gee",
    "h": "hee",
    "i": "i",
    "j": "jee",
    "k": "kee",
    "l": "lee",
    "m": "mee",
    "n": "nee",
    "o": "o",
    "p": "pee",
    "q": "qee",
    "r": "ree",
    "s": "see",
    "t": "tee",
    "u": "u",
    "v": "vee",
    "x": "xee",
    "y": "yee",
    "z": "zee",
    "g'": "g'ee",
    "o'": "o'",
    "sh": "shee",
    "ch": "chee",
    "ng": "ning",
    "w": "vee",
    "c":"si",
    ",":",",
    ".":"."
}

def replace_word(word):
  output_text=str()
  for latter in word:
           output_text+=str(vocab_uzb[latter.lower()])
  return output_text

def regenerate_text(text):
  response_text=str()
  for word in text.split(" "):
      if len(word)<=5 and len(word)>=3 and word[0]==word[0].capitalize() and word[-1]==word[-1].capitalize():
        word=replace_word(word)
      response_text+=str(word)+" "
  return response_text

def check_upper(text):
    if text == text.upper() and len(text) > 6:
        return text.lower()
    return text

def dupicate_i(text):
    return text.replace("i", "ii")

def uzbek_cleaners(text):

    text = replace_s(text=text)
    text = transliterate(text, "latin")
    text = expand_numbers(text)
    text = re.sub(_websites_re, replace_website_symbols, text)
    text = re.sub(_main_symbol_re, replace_main_symbols, text)
    text = replace_apostrophe(text)
    text = collapse_symbols(text)
    text = re.sub(_c_symbol_re, replace_c, text)
    text = collapse_whitespace(text)
    text = non_uzbek(text)
    text = add_fullstop(text)
    text = collapse_fullstop(text)
    text = collapse_newline(text)
    text = re.sub(_combined, i_character, text, flags=re.IGNORECASE)
    text = check_upper(text)
    # text = regenerate_text(text=text)
    text = lowercase(text)
    text = re.sub(_wtf_re, ". ", text)
    text = remove_disallowed_chars(text=text)
    #text = dupicate_i(text)

    return text

