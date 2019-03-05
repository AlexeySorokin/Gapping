import numpy as np
import bisect
import copy

from udapi.core.node import Node

WORD_COLUMN, LEMMA_COLUMN, POS_COLUMN, FEAT_COLUMN, HEAD_COLUMN, REL_COLUMN = 1, 2, 3, 5, 6, 7
HYPHENS = "-—–"
QUOTES = "«“”„»``''\""
# data format manipulations

def get_start_positions(text, words):
    starts = []
    start = 0
    for word in words:
        while not text[start:].startswith(word):
            if text[start] in QUOTES and word in QUOTES:
                break
            start += 1
        starts.append(start)
        start += len(word) if word not in QUOTES else 1
    assert len(starts) == len(words)
    return starts


def char_to_word_positions(sent, words, positions):
    positions = np.array(positions, dtype=int)
    flat_positions = np.ravel(positions)
    start_positions = get_start_positions(sent, words)
    word_positions = [bisect.bisect_left(start_positions, x) if x != -1 else -1
                      for x in flat_positions]
    return np.reshape(word_positions, positions.shape).tolist()


def get_tag(elem):
    return (elem.upos, dict(elem.feats))


def tag_str(elem):
    pos, feats = elem.upos, str(elem.feats)
    if feats == "_":
        return pos
    else:
        return "{} {}".format(pos, feats)

# word manipulations

def is_possible_as_remnant(word):
    return all(x.isalpha() or x.isdigit() or x in "-/." for x in word) or word == "%"

def is_word(word):
    word = word.strip(".…")
    if word in "%$":
        return True
    if word == "" or not any((x.isalpha() or x.isdigit()) for x in word):
        return False
    return all(x.isalpha() or x.isdigit() or x in "-/.,,—*" or x in HYPHENS for x in word)

# tag manipulations

NOUN, ADJ, CONJ = ["NOUN", "PROPN", "PRON"], ["ADJ", "DET"], ["CCONJ", "SCONJ"]

def is_amod(node):
    return node.deprel in ["det", "amod", "nummod:gov"]

def is_dependent_inf(elem):
    has_aux = any(child.deprel == "aux" for child in elem.children)
    return elem.feats.get("VerbForm") == "Inf" and not has_aux and elem.parent.ord != 0

def is_verb(elem, allow_part=True, allow_short=True, allow_inf=True):
    if elem.upos not in ["VERB", "AUX"] and (not allow_short or not is_short(elem)):
        return False
    if not allow_inf and is_dependent_inf(elem):
        return False
    if allow_part or elem.feats.get("VerbForm") != "Part":
        return True
    return allow_short and elem.feats.get("Variant") == "Short"

def is_participle(elem, variant=None):
    if elem.upos != "VERB" or elem.feats.get("VerbForm") != "Part":
        return False
    if variant == "full" and elem.feats.get("Variant") == "Short":
        return False
    if variant == "short" and elem.feats.get("Variant", "Short") != "Short":
        return False
    return True

def is_short(elem):
    return elem.upos == "ADJ" and elem.feats.get("Variant") == "Short"

def is_copula(elem, only_adj=True):
    return (elem.upos == "ADJ" or not only_adj) and any(
        (child.deprel == "cop" and child.feats.get("VerbForm") != "Inf") for child in elem.children)

def is_adj(elem, allow_det=True, allow_part=False, allow_short=True, allow_num=True):
    adj_pos = ["ADJ", "DET"] if allow_det else "ADJ"
    answer = elem.upos in adj_pos
    if allow_part:
        variant = "full" if not allow_part else None
        answer |= is_participle(elem, variant=variant)
    if allow_num:
        answer |= (elem.upos == "NUM" and ("nummod" in elem.deprel or elem.parent.upos == "VERB"))
    if not allow_short:
        answer &= not(elem.get("Variant") != "Short")
    return answer

FEAT_KEYS = ["Number", "Gender", "Case"]

def is_adj_noun_match(adj_node, noun_node, check_adj=False, check_noun=False, allow_part=True):
    if check_adj and not is_adj(adj_node, allow_part=allow_part):
        return False
    if check_noun and not is_noun(noun_node):
        return False
    adj_number, adj_gender, adj_case = [adj_node.feats.get(key) for key in FEAT_KEYS]
    noun_number, noun_gender, noun_case = [noun_node.feats.get(key) for key in FEAT_KEYS]
    answer = (noun_case == adj_case) or noun_case in ["", None]
    answer &= (noun_number == adj_number) or noun_number in ["", None] or adj_number in ["", None]
    answer &= (noun_number == "Plur") or (noun_gender == adj_gender) or (noun_gender in ["", None])
    return answer


def is_verb_noun_match(noun_node, verb_node, allow_short=False):
    if not is_verb(verb_node, allow_inf=False, allow_part=False, allow_short=allow_short):
        return False
    noun_number, noun_gender = noun_node.feats.get("Number"), noun_node.feats.get("Gender")
    verb_number, verb_gender = verb_node.feats.get("Number"), verb_node.feats.get("Gender")
    if noun_node.feats.get("Case") not in ["Nom", None, ""]:
        return False
    answer = noun_number == "" or noun_number == verb_number
    answer &= noun_gender == "" or verb_gender == "" or verb_gender == noun_gender
    return answer

AMBIGIOUS_PRONOUNS = ["её", "ее", "его", "их"]

def is_noun(elem):
    return elem.upos in ["NOUN", "PROPN"] or\
           (elem.upos == "PRON" and (elem.form not in AMBIGIOUS_PRONOUNS or elem.deprel != "nmod"))


def have_equal_pos(first, second, allow_part=False):
    return (are_equal_pos(first.upos, second.upos) or
            (is_adj(first, allow_part=allow_part) and is_adj(second, allow_part=allow_part)) or
            (is_noun(first) and is_noun(second)))

def are_equal_pos(first, second):
    return first == second or any((first in s and second in s) for s in [ADJ, CONJ, NOUN])


def have_equal_case(first, second, prep=False, check_equal_pos=False, allow_part=False):
    if check_equal_pos and not have_equal_pos(first, second, allow_part=allow_part):
        return False
    answer = (first.feats.get("Case") == second.feats.get("Case"))
    if prep:
        first_prep, second_prep = extract_prep(first, allow_conj=False), extract_prep(second, allow_conj=False)
        answer &= (first_prep == second_prep)
    return answer

# tree manipulations

SPECIAL_WORDS = ["ни", "."]


def DEPREL_CHECKER(node: Node):
    if node.deprel != "conj":
        return True
    children = node.children(preceding_only=True)
    child_deprel = children[0].deprel if len(children) > 0 else None
    return have_equal_pos(node, node.parent) and have_equal_case(node, node.parent) and child_deprel in ["cc", "punct"]


def has_num_dep(node):
    return any(child.upos == "NUM" for child in node.children(preceding_only=True))


def extract_prep(node: Node, allow_conj: bool = True, return_form=True):
    children = list(node.children(preceding_only=True)[::-1])
    if allow_conj:
        parent, deprel = node.parent, node.deprel
        if deprel == "conj" and have_equal_pos(parent, node) and have_equal_case(parent, node):
            children += list(parent.children(preceding_only=True)[::-1])
    while len(children) > 0:
        child = children.pop()
        if child.upos == "ADP" and child.deprel == "case":
            return child.form.lower() if return_form else child
    return None

def have_equal_prep(first, second, allow_conj=False):
    first_prep = extract_prep(first, allow_conj=allow_conj)
    second_prep = extract_prep(second, allow_conj=allow_conj)
    if first_prep is not None and second_prep is not None:
        return first_prep.rsplit("о") == second_prep.rsplit("о")
    return first_prep == second_prep


def make_possible_tags(node):
    if is_noun(node):
        possible_cases = get_possible_noun_cases(node, change_number=is_noun(node), return_number=True)
    elif is_adj(node) :#and not is_amod(node):
        possible_cases = get_possible_adj_features(node)
    else:
        possible_cases = []
    case, number = node.feats["Case"], node.feats["Number"]
    answer = []
    for elem in possible_cases:
        other_case, other_number, other_gender = list(elem[:2]) + [None]
        if len(elem) == 3:
            other_gender = elem[2]
        if other_case == case:
            continue
        new_node = copy.deepcopy(node)
        new_node.feats["Case"] = other_case
        new_node.feats["Number"] = other_number
        answer.append(new_node)
    return answer

def get_possible_adj_features(node):
    keys = ["Animacy", "Number", "Gender", "Case"]
    pos = node.upos
    animacy, number, gender, case = [node.feats[key] for key in keys]
    answer = [(case, number, gender)]
    if number == "Sing":
        if gender == "Fem":
            confused_cases = ["Gen", "Dat", "Ins", "Loc"]
            if case in confused_cases:
                answer = [(other, number, gender) for other in confused_cases]
        else:
            confused_cases = ["Acc", "Gen" if animacy == "Anim" or pos != "ADJ" else "Nom"]
            if case in confused_cases:
                answer = [(other, number, gender) for other in confused_cases]
            if case == "Ins":
                answer.append(("Dat", "Plur", ""))
        if node.form.endswith("ой"):
            answer += [(other, "Sing", "Fem") for other in ["Gen", "Dat", "Ins", "Loc"]]
    else:
        confused_cases = ["Gen", "Loc"]
        if animacy == "Anim":
            confused_cases.append("Acc")
        if case in confused_cases:
            answer = [(other, number, gender) for other in confused_cases]
        if animacy == "Inan":
            confused_cases = ["Nom", "Acc"]
            if case in confused_cases:
                answer = [(other, number, gender) for other in confused_cases]
        if case == "Dat":
            answer.extend([("Dat", "Sing", "Masc"), ("Dat", "Sing", "Fem")])
    return answer

def get_possible_noun_cases(node, change_number=False, return_number=False):
    keys = ["Animacy", "Number", "Gender", "Case"]
    pos = node.upos
    values = animacy, number, gender, case = [node.feats[key] for key in keys]
    answer = [case]
    if all(x is not None for x in values):
        if pos in ["NOUN", "PROPN", "PRON"]:
            second_acc = "Gen" if (animacy == "Anim" or pos == "PRON") else "Nom"
            if gender in ["Masc", "Neut"] or number == "Plur" or node.lemma[-1] == "ь" or pos == "PRON":
                if (animacy == "Anim" or pos == "PRON") and case == "Gen":
                    answer = ["Gen", "Acc"]
                if animacy == "Inan" and case == "Nom" and pos != "PRON":
                    if "nsubj" not in node.deprel or node.parent.upos not in ["VERB", "AUX"]:
                        answer = ["Nom", "Acc"]
                if case == "Acc":
                    answer.append(second_acc)
            if (gender == "Fem" and number == "Sing" and animacy == "Inan"):
                confused_cases = ["Dat", "Loc"]
                if node.lemma[-1] == "ь":
                    confused_cases += ["Gen"]
                if case in confused_cases:
                    answer += confused_cases
    answer = [(x, number) for x in set(answer)]
    if change_number:
        if pos == "NOUN" and gender in ["Fem", "Neut"]:
            if number == "Sing" and case == "Gen":
                answer.append(("Nom", "Plur"))
                if animacy == "Inan":
                    answer.append(("Acc", "Plur"))
            if gender == "Fem" and number == "Sing" and animacy == "Inan" and node.lemma[-1] == "ь":
                if case in ["Dat", "Loc"]:
                    answer.append(("Nom", "Plur"))
                    answer.append(("Acc", "Plur"))
            if number == "Plur":
                if ("nsubj" not in node.deprel or node.parent.upos not in ["VERB", "AUX"]) and case == "Nom":
                    answer.append(("Gen", "Sing"))
                if animacy == "Inan" and case == "Acc":
                    answer.append(("Gen", "Sing"))
    if any(child.upos == "ADP" for child in node.children):
        answer = [(x, y) for x, y in answer if x != "Nom"]
    if not return_number:
        answer = [elem[0] for elem in answer]
    return answer


def can_have_equal_case(first, second):
    if is_noun(first) and is_noun(second):
        first_cases = get_possible_noun_cases(first)
        second_cases = get_possible_noun_cases(second)
        return any(x in second_cases for x in first_cases)
    return False


# def do_nodes_match(first: Node, second: Node):
#     first_prep, second_prep = extract_prep(first), extract_prep(second)
#     match = int(have_equal_pos(first, second) and have_equal_case(first, second))
#     if match == 1:
#         match += 1 + int(first_prep == second_prep)
#     else:
#         match = int(can_have_equal_case(first, second))
#     return match


def normalize_node(node, match_rel=None):
    answer = node
    if is_adj(node):
        if (extract_prep(node.parent) or match_rel == "nsubj") and is_noun(node.parent):
            answer = node.parent
    if is_copula(node, only_adj=False) and match_rel is None:
        children = [child for child in node.children if child.deprel == "cop"]
        if len(children) > 0:
            answer = children[0]
    elif node.deprel == "fixed":
        answer = node.parent
    elif node.deprel == "nummod:gov" and not is_verb(node.parent):
        answer = node.parent
    elif node.deprel in ["xcomp", "csubj"] and node.feats.get("VerbForm") == "Inf":
        answer = node.parent
    return answer


def check_parataxis(node):
    siblings = [node.parent.children, node.children]
    if node.deprel != "parataxis":
        return False
    for elem in siblings:
        if (len(elem) >= 2 and elem[0].ord < node.ord and elem[-1].ord > node.ord):
            if (elem[0].form + elem[-1].form) in [",,", "()"]:
                return True
    return False


def can_be_second_in_gap(node, orphan_rel):
    if orphan_rel != "nmod":
        return True
    return node.feats.get("Case") != "Gen" or any(child.upos == "ADP" for child in node.children)


def can_be_verb(node, allow_inf=True):
    return (is_verb(node, allow_part=False, allow_inf=allow_inf) or is_copula(node, only_adj=False)
                or is_adj(node) and node.feats.get("Variant") == "Short" and node.parent.ord == 0)