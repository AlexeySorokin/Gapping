import os
from collections import defaultdict

from udapi.core.node import Node

from common import *
from common import HYPHENS, is_word
from read_write import read_data, read_parse_file
from gap_types import make_phrase_repr
from fix_tree import fix_tree

from udapi.core.document import Document
from udapi.block.write.textmodetrees import TextModeTrees

from subtree import find_subtree_spans, to_subtree_heads


def char_to_word_labels(source_sents, word_sents, labels):
    answer = []
    for r, (source_sent, sent, curr_labels) in enumerate(zip(source_sents, word_sents, labels), 1):
        if curr_labels is not None:
            curr_labels = char_to_word_positions(source_sent, sent, curr_labels)
        answer.append(curr_labels)
    return answer


def do_nodes_match(first, second, allow_fuzzy_match=True):
    if second.lemma in ["%", "процент", "раз"]:
        return first.lemma in ["%", "процент", "раз"]
    if first.upos == "ADV" and second.upos == "ADJ" and second.feats["Degree"] == "Cmp":
        return True
    if second.upos == "ADV" and first.upos == "ADJ" and first.feats["Degree"] == "Cmp":
        return True
    if is_adj(second, allow_part=True):
        answer = have_equal_case(first, second, check_equal_pos=True, allow_part=True)
        if is_amod(first):
            first_prep, second_prep = extract_prep(first.parent, allow_conj=False), extract_prep(second, allow_conj=False)
            answer &= (first_prep is not None) == (second_prep is not None)
    else:
        answer = have_equal_case(first, second, check_equal_pos=True)
        first_prep, second_prep = extract_prep(first, allow_conj=False), extract_prep(second, allow_conj=False)
        if (first.upos != "NUM" and second.upos != "NUM") and not (has_num_dep(first) and has_num_dep(second)):
            answer &= ((first_prep is not None) == (second_prep is not None))
    fuzzy_match = 0
    if not answer and have_equal_pos(first, second, allow_part=True) and allow_fuzzy_match:
        possible_nodes = make_possible_tags(second)
        if len(possible_nodes) > 0:
            fuzzy_match = max((do_nodes_match(first, other, allow_fuzzy_match=False)
                               for other in possible_nodes))
        else:
            possible_nodes = make_possible_tags(first)
            if len(possible_nodes) > 0:
                fuzzy_match = max((do_nodes_match(other, second, allow_fuzzy_match=False)
                                   for other in possible_nodes)) - 1
    if answer:
        return 6
    elif fuzzy_match > 1:
        return max(fuzzy_match-1, 0)
    elif is_adj(second) and is_noun(first) and have_equal_case(first, second, prep=True):
        return 3
    elif is_noun(second) and is_adj(first) and have_equal_case(first, second, prep=True):
        return 3
    else:
        return 0

def has_matching_forms(first, second):
    return (first.lemma == second.lemma) or (first.upos == "NUM" == second.upos == "NUM")

def select_match(candidates, anchor, verb, side):
    if len(candidates) == 1:
        return candidates[0][0], candidates, [None]
    elif len(candidates) == 0:
        return None, [], []
    candidate_keys = [(has_matching_forms(node, anchor),
                       int(node.ord > verb.ord) == side,
                       level >= 6, -depth, level,
                       has_num_dep(node) == has_num_dep(anchor),
                       have_equal_prep(node, anchor),
                       node.form.lower() not in ["то", "это"],
                      ) for node, depth, level in candidates]
    order = sorted(enumerate(candidate_keys), key=lambda x: x[1], reverse=True)
    order, ordered_keys = [elem[0] for elem in order], [elem[1] for elem in order]
    candidates = [candidates[i] for i in order]
    if ordered_keys[0] > ordered_keys[1]:
        answer = candidates[0][0]
    else:
        answer = None
    return answer, candidates, ordered_keys

def normalize_matching_node(node, anchor):
    if is_amod(node) and node.deprel != "nummod:gov":
        if is_adj_noun_match(node, node.parent, check_adj=True, check_noun=True):
            # TO INVESTIGATE BETTER
            node = node.parent
    elif node.deprel == "nummod:gov":
        node = node.parent
    elif node.deprel == "nummod":
        if anchor.upos != "NUM" and is_noun(node.parent):
            node = node.parent
    return node

def find_remnant(sent, verb, left, right, break_ties=False):
    sent = sent.descendants
    verb, left, right = sent[verb], sent[left], sent[right]
    candidates = []
    if verb.deprel == "cop" and verb.parent.ord > 0:
        verb = verb.parent
        candidates.append((verb, 0))
    verb_children = list(verb.children)
    for child in verb.children(following_only=True):
        if child.feats.get("VerbForm") == "Inf":
            verb_children += child.children
            verb_children.sort(key=(lambda x: x.ord))
            break
    for child in verb_children:
        child_descendants = child.descendants(add_self=True)
        # if child_descendants.ord >= left.ord:
        if child.ord >= left.ord:
            break
        if child.ord > verb.ord:
            if child.deprel == "conj":
                break
            if child.deprel in ["obl"]:
                child_children = child.children(preceding_only=True)
                if len(child_children) > 0:
                    if child_children[0].form == ",":
                        break
        candidates.append((child, 0))
    candidates_queue = candidates[:]
    while len(candidates_queue) > 0:
        node, k = candidates_queue.pop()
        for other in node.children(preceding_only=True):
            if is_amod(other) :#and is_adj_noun_match(other, node):
                candidates.append((other, k+1))
            if other.deprel == "nummod":
                candidates.append((other, k + 1))
        for other in node.children(following_only=True):
            if other.ord >= left.ord:
                break
            if other.deprel == "nummod":
                candidates.append((other, k + 1))
            if other.deprel in ["nmod", "obl"]:
                candidates.append((other, k + 1))
                candidates_queue.append((other, k + 1))
            if other.deprel == "conj" and node.deprel == "nmod" and other.feats["Case"] == node.feats["Case"]:
                candidates.append((other, k+0.1))
                candidates_queue.append((other, k+0.1))
    left_candidates, right_candidates = [], []
    for node, level in candidates:
        match = do_nodes_match(node, left)
        if match > 0:
            left_candidates.append((node, level, match))
        match = do_nodes_match(node, right)
        if match > 0:
            right_candidates.append((node, level, match))
    remnants = None
    left_match, left_candidates, left_keys = select_match(left_candidates, left, verb, 0)
    right_match, right_candidates, right_keys = select_match(right_candidates, right, verb, 1)
    if right_match is not None and right_match.ord > verb.ord:
        indexes = [i for i, node in enumerate(left_candidates) if node[0].ord < right_match.ord]
        if 0 not in indexes:
            left_match = None
        left_candidates = [left_candidates[i] for i in indexes]
        left_keys = [left_keys[i] for i in indexes]
        if left_match is None or left_match.ord >= right_match.ord:
            indexes = [i for i, elem in enumerate(left_candidates) if elem[0].ord < right_match.ord]
            left_candidates = [left_candidates[i] for i in indexes]
            left_keys = [left_keys[i] for i in indexes]
            # if len(indexes) > 0:
            #     index = indexes[0]
            #     left_candidates = left_candidates[:index] + left_candidates[index+1:]
            #     left_keys = left_keys[:index] + left_keys[index+1:]
        if len(left_keys) > 0 and (len(left_keys) == 1 or left_keys[0] > left_keys[1] or break_ties):
            left_match = left_candidates[0][0]
    if left_match is not None and right_match is not None:
        if left_match.ord < right_match.ord:
            remnants = [normalize_matching_node(left_match, left),
                        normalize_matching_node(right_match, right)]
    if remnants is not None:
        return [x.ord - 1 for x in remnants]
    return None, None


def get_extraction_key(corr, pred):
    if pred is None:
        return "parse error"
    elif pred[0] is None:
        return "FN"
    corr_number, err_number = 0, 0
    for x, y in zip(corr[1:], pred):
        if x == y:
            corr_number += 1
        elif x != y and x is not None:
            err_number += 1
    if corr_number > 0:
        return "TP" if err_number == 0 else "partial"
    else:
        return "wrong" if err_number > 0 else "parse error"

def get_subtree_extraction_key(corr, pred):
    if pred is None or pred[0] is None:
        return "FN"
    corr_number, err_number = 0, 0
    for x, y in zip(corr[1:], map(list, pred)):
        if x == y:
            corr_number += 1
        elif x != y and x is not None:
            err_number += 1
    if corr_number > 0:
        return "TP" if err_number == 0 else "partial"
    else:
        return "wrong" if err_number > 0 else "parse error"

def output_matching_results(outfile, head_answer, subtree_labels,
                            span_answer, span_labels, sents, parsed_sents,
                            indexes=None, output_trees=False, output_correct=False,
                            compare_with_first=True):
    if indexes is None:
        indexes = [(i, j) for i, elem in enumerate(subtree_labels) for j in range(len(elem)-1)]
    arranged_indexes = defaultdict(list)
    for i, j in indexes:
        arranged_indexes[i].append(j)
    arranged_indexes = sorted(arranged_indexes.items())
    with open(outfile, "w", encoding="utf8") as fout:
        for i, curr_indexes in arranged_indexes:
            curr_subtree_labels, curr_subtree_spans = subtree_labels[i], span_labels[i]
            writer = TextModeTrees(filehandle=fout, attributes="form,upos,deprel,ord,feats")
            sent = parsed_sents[i].descendants
            fout.write("{}\t{}\n{}\n".format(i, sents[i], "=" * 40))
            if output_trees:
                writer.before_process_document(Document())
                writer.process_tree(parsed_sents[i])
                writer.after_process_document(Document())
                fout.write("=" * 40 + "\n")
            for j in curr_indexes:
                nodes = [curr_subtree_labels[0][0], curr_subtree_labels[j+1][1], curr_subtree_labels[j+1][2]]
                nodes = [sent[k] if k is not None else None for k in nodes]
                fout.write("\t".join("{}-{}".format(node.ord, node.form) if node is not None else "None"
                                     for node in nodes))
                fout.write("\t|\t")
                if head_answer[i][j] is not None:
                    nodes = [sent[k] if k is not None else None for k in head_answer[i][j]]
                    fout.write("\t".join("{}-{}".format(node.ord, node.form) if node is not None else "None"
                                         for node in nodes))
                else:
                    fout.write("None\tNone")
                if output_correct:
                    fout.write("\t|\t")
                    nodes = [sent[k] if k is not None else None for k in curr_subtree_labels[0][1:]]
                    fout.write("\t".join("{}-{}".format(node.ord, node.form) if node is not None else "None"
                                         for node in nodes))
                fout.write("\n" + "-" * 40 + "\n")
                if span_answer[i][j] is not None:
                    left_phrase_repr = make_phrase_repr(sent, *span_answer[i][j][0])
                    right_phrase_repr = make_phrase_repr(sent,*span_answer[i][j][1])
                else:
                    left_phrase_repr, right_phrase_repr = None, None
                fout.write("{}\t{}".format(left_phrase_repr, right_phrase_repr))
                if output_correct:
                    fout.write("\t|\t")
                    j_corr = 0 if compare_with_first else j + 1
                    left_phrase_repr = make_phrase_repr(sent, *curr_subtree_spans[j_corr][1])
                    right_phrase_repr = make_phrase_repr(sent, *curr_subtree_spans[j_corr][2])
                    fout.write("{}\t{}".format(left_phrase_repr, right_phrase_repr))
                fout.write("\n\n")
            # if len(answer[i]) > 0:
            #     for elem in answer[i]:
            #         nodes = [sent[j] for j in elem]
            #         fout.write("\t".join("{}-{}".format(node.ord, node.form) for node in nodes) + "\n")
            # if output_correct and corr_answer[i] is not None:
            #     fout.write("=" * 15 + " CORRECT " + "=" * 15 + "\n")
            #     for elem in corr_answer[i]:
            #         if elem == "error":
            #             fout.write("error\n")
            #         else:
            #             nodes = [sent[j] for j in elem]
            #             fout.write("\t".join("{}-{}".format(node.ord, node.form) for node in nodes) + "\n")
            fout.write("\n")




if __name__ == "__main__":
    tree_fixes = ["fix_adv_adj"]
    source_file, infile = "data/train.csv", "results/example_1.out"
    output_dir, model_name = "match_results_4000", "basic.adj.nmod.cop.an.homo.ties"
    (source_sents, labels), parsed_sents = read_data(source_file), read_parse_file(infile, max_sents=4000, parse=False)
    word_sents = [[elem.form for elem in sent.descendants] for sent in parsed_sents]
    word_labels = char_to_word_labels(source_sents, word_sents, labels)
    subtree_labels = to_subtree_heads(parsed_sents, word_labels)
    subtree_heads, subtree_spans = [], []
    for i, (sent, curr_labels) in enumerate(zip(parsed_sents, subtree_labels)):
        if len(tree_fixes) > 0:
            curr_fixes = fix_tree(sent, tree_fixes)
        if curr_labels is None:
            subtree_heads.append([])
            subtree_spans.append([])
            continue
        curr_subtree_heads, curr_subtree_spans = [], []
        verb = curr_labels[0][0]
        try:
            leftmost = min(elem[1] for elem in curr_labels[1:] if elem[1] is not None)
        except:
            leftmost = None
        for _, left, right in curr_labels[1:]:
            if left is not None and right is not None:
                left_head, right_head = find_remnant(sent, verb, left, right)
                if left_head is not None and right_head is not None:
                    spans = find_subtree_spans(sent, verb, left_head, right_head, leftmost)
                else:
                    spans = None
                curr_subtree_heads.append((left_head, right_head))
                curr_subtree_spans.append(spans)
            else:
                curr_subtree_heads.append(None)
                curr_subtree_spans.append(None)
        subtree_heads.append(curr_subtree_heads)
        subtree_spans.append(curr_subtree_spans)
    stats = defaultdict(list)
    for i, (corr, pred) in enumerate(zip(subtree_labels, subtree_heads)):
        if corr is None:
            continue
        for j, elem in enumerate(pred):
            key = get_extraction_key(corr[0], elem)
            subtree_key = get_subtree_extraction_key(word_labels[i][0], subtree_spans[i][j])
            joint_key = ("{}_{}".format(key, subtree_key))
            stats[joint_key].append((i, j))
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    KEYS = ["TP_TP", "TP_partial", "TP_wrong", "partial_partial", "partial_wrong", "FN_FN"]
    for key in KEYS:
        print(key, len(stats[key]))
        outfile = os.path.join(output_dir, key)
        output_matching_results(outfile, subtree_heads, subtree_labels,
                                subtree_spans, word_labels,
                                source_sents, parsed_sents, stats[key],
                                output_trees=(key != "TP_TP"),
                                output_correct=(key != "TP_TP"))
    # for start in range(0, len(stats["partial"]), 20):
    #     print(*("{},{}".format(*x) for x in stats["partial"][start:start+20]))


def find_gap_position(sent, left, right):
    if isinstance(sent, Node):
        sent = sent.descendants
    left_node, right_node = sent[left], sent[right]
    left_descendants = left_node.descendants(add_self=True)
    right_descendants = right_node.descendants(add_self=True)
    if right_descendants[0].ord <= left + 1:
        # left -- зависимый right
        answer = left_descendants[-1].ord
    else:
        answer = right_descendants[0].ord - 1
    if answer == len(sent):
        return None
    if sent[answer].form in HYPHENS:
        answer += 1
    new_answer = answer
    # while new_answer < len(sent) and not is_word(sent[new_answer].form):
    #     new_answer += 1
    while True:
        if new_answer == len(sent):
            return answer
        parent = sent[new_answer].parent
        siblings = parent.descendants(add_self=True)
        if parent.deprel == "parataxis" and siblings[-1].ord-1 < right:
            new_answer = siblings[-1].ord
        elif not is_word(sent[new_answer].form):
            new_answer = new_answer + 1
        else:
            break
    return new_answer