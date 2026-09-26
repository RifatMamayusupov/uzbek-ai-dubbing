import re

import inflect

_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(-)")
_number_re = re.compile(r"[0-9]+")

ones = {
    0: '', 1: 'bir', 2: 'ikki', 3: 'uch', 4: 'to‘rt', 5: 'besh', 6: 'olti',
    7: 'yetti', 8: 'sakkiz', 9: 'to‘qqiz', 10: 'o‘n', 11: 'o‘n bir', 12: 'o‘n ikki',
    13: 'o‘n uch', 14: 'o‘n to‘rt', 15: 'o‘n besh', 16: 'o‘n olti',
    17: 'o‘n yetti', 18: 'o‘n sakkiz', 19: 'o‘n to‘qqiz'}

tens = {
    2: 'yigirma', 3: 'o‘ttiz', 4: 'qirq', 5: 'ellik', 6: 'oltmish',
    7: 'yetmish', 8: 'sakson', 9: 'to‘qson'}

illions = {
    1: 'ming', 2: 'million', 3: 'milliard', 4: 'trillion', 5: 'kvadrillion',
    6: 'kvintillion', 7: 'sekstillion', 8: 'septillion', 9: 'oktillion',
    10: 'nonillion', 11: 'detsillion'}


def say_number(i):
    if i < 0:
        return _join('manfiy', _say_number_pos(-i))
    if i == 0:
        return 'nol'
    return _say_number_pos(i)


def _say_number_pos(i):
    if i < 20:
        return ones[i]
    if i < 100:
        return _join(tens[i // 10], ones[i % 10])
    if i < 1000:
        return _divide(i, 100, 'yuz')
    for illions_number, illions_name in illions.items():
        if i < 1000**(illions_number + 1):
            break
    return _divide(i, 1000**illions_number, illions_name)


def _divide(dividend, divisor, magnitude):
    return _join(
        _say_number_pos(dividend // divisor),
        magnitude,
        _say_number_pos(dividend % divisor),
    )


def _join(*args):
    return ' '.join(filter(bool, args))


def _remove_commas(m):
    return m.group(1).replace(",", ",")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " nuqta ")


def _expand_ordinal(text):

    text = re.sub(r"(\d)\s*-\s*", r"\1inchi ", text)
    return text


def _expand_number(m):
    
    num = int(m.group(0))
    if m[0][0]=="0" and len(m[0])==2:
       res = "nol "+say_number(num)
       return res
    elif m[0][0]=="0" and m[0][1]=="0" and len(m[0])==3:
       res = "nol " + "nol " + say_number(num)
       return res
    elif m[0][0]=="0" and len(m[0])==3:
       res = "nol "+say_number(num)
       return res

    return say_number(num)

def expend_numbers(text):

    # text = replace_roman_with_digits(text=text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = _expand_ordinal(text=text)
    text = re.sub(_number_re, _expand_number, text)

    return text

def find_hours_in_text(text):

    pattern = r"\b(\d{1,2}):(\d{2})(\s+|)?"
    
    def replace_with_words(match):
        hour, minute, suffix = match.groups()
        hour_word = expend_numbers(hour)
        minute_word = expend_numbers(minute)
        # Build the return string, including suffix if it exists
        result = f"{hour_word}u {minute_word}"
        if suffix:
            result += f" {suffix}"
        return result

    modified_text = re.sub(pattern, replace_with_words, text)
    return modified_text

def find_phone_numbers(text):
    
    pattern = r"\+998\d{9}|998\d{9}|\d{9}|\d{3} \d{2} \d{3} \d{2} \d{2}|\d{2} \d{3} \d{2} \d{2}"
    matches = re.findall(pattern, text)
    return matches


def expand_phone_number(text):
    for phone in find_phone_numbers(text):
        normalized_phone = "".join(phone.split())

        if len(normalized_phone) == 13: 
            expanded_num = ", ".join([
                normalized_phone[:1], normalized_phone[1:4], normalized_phone[4:6], 
                normalized_phone[6:9], normalized_phone[9:11], normalized_phone[11:]
            ])+ ","
        elif len(normalized_phone) == 12:  
            expanded_num = ", ".join([
                normalized_phone[:3], normalized_phone[3:5], normalized_phone[5:8], 
                normalized_phone[8:10], normalized_phone[10:]
            ]) + ","
        elif len(normalized_phone) == 9:  
            expanded_num = ", ".join([
                normalized_phone[:2], normalized_phone[2:5], 
                normalized_phone[5:7], normalized_phone[7:]
            ]) + ","
        else:
            continue  

        text = text.replace(phone, expanded_num)
    
    return text

def normalize_numbers(text):
   
    text = find_hours_in_text(text=text)
    text = expand_phone_number(text=text)
    text = expend_numbers(text=text)
    text = text.replace("bir yuz", "yuz")

    return text
