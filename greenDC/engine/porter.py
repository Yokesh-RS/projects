# Porter stemmer, following M.F. Porter, "An algorithm for suffix stripping" (1980).

STEP2_RULES = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")
]

STEP3_RULES = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", "")
]

STEP4_SUFFIXES = [
    "ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ion",
    "ism", "ate", "iti", "ous", "ive", "ize", "al", "er", "ic", "ou"
]

cache = {}


def is_consonant(word, i):
    if word[i] in "aeiou":
        return False
    if word[i] == "y":
        return i == 0 or not is_consonant(word, i - 1)
    return True


def measure(stem):
    # Number of vowel-consonant sequences, the "m" in Porter's rules.
    pattern = ""
    for i in range(len(stem)):
        pattern += "c" if is_consonant(stem, i) else "v"
    return pattern.count("vc")


def has_vowel(stem):
    return any(not is_consonant(stem, i) for i in range(len(stem)))


def ends_double_consonant(word):
    return len(word) >= 2 and word[-1] == word[-2] and is_consonant(word, len(word) - 1)


def ends_cvc(word):
    if len(word) < 3 or word[-1] in "wxy":
        return False
    n = len(word)
    return is_consonant(word, n - 3) and not is_consonant(word, n - 2) and is_consonant(word, n - 1)


def apply_rules(word, rules):
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if measure(stem) > 0:
                return stem + replacement
            return word
    return word


def step1(word):
    if word.endswith("sses") or word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]

    if word.endswith("eed"):
        if measure(word[:-3]) > 0:
            word = word[:-1]
    else:
        for suffix in ("ed", "ing"):
            if word.endswith(suffix) and has_vowel(word[:-len(suffix)]):
                word = word[:-len(suffix)]
                if word.endswith(("at", "bl", "iz")):
                    word += "e"
                elif ends_double_consonant(word) and word[-1] not in "lsz":
                    word = word[:-1]
                elif measure(word) == 1 and ends_cvc(word):
                    word += "e"
                break

    if word.endswith("y") and has_vowel(word[:-1]):
        word = word[:-1] + "i"
    return word


def step4(word):
    for suffix in STEP4_SUFFIXES:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if measure(stem) > 1:
                if suffix == "ion" and not stem.endswith(("s", "t")):
                    return word
                return stem
            return word
    return word


def step5(word):
    if word.endswith("e"):
        m = measure(word[:-1])
        if m > 1 or (m == 1 and not ends_cvc(word[:-1])):
            word = word[:-1]
    if word.endswith("ll") and measure(word) > 1:
        word = word[:-1]
    return word


def stem(word):
    if len(word) <= 2:
        return word
    if word not in cache:
        result = step1(word)
        result = apply_rules(result, STEP2_RULES)
        result = apply_rules(result, STEP3_RULES)
        result = step4(result)
        cache[word] = step5(result)
    return cache[word]
