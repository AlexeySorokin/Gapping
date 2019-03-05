import os
from io import StringIO
from collections import defaultdict
from functools import partial

from common import *
from read_write import read_data, read_parse_file, output_results

from udapi.block.read.conllu import Conllu
from udapi.core.node import Node, ListOfNodes

from subtree import get_subtree_bounds

GAP_RELATIONS = ["orphan", "nsubj", "parataxis", "appos", "nmod"]

def orphan(sent: ListOfNodes, orphan_rel="orphan", use_blocking_relations=True):
    # V -> NOUN -> orphan
    head_gap_pairs = []
    if use_blocking_relations:
        try:
            index = GAP_RELATIONS.index(orphan_rel)
        except ValueError:
            index = len(GAP_RELATIONS)
        blocking_relations = GAP_RELATIONS[:index]
    else:
        blocking_relations = []
    processed_heads = set()
    for i, elem in enumerate(sent):
        if not (is_verb(elem) or is_short(elem) or is_copula(elem)) or elem in processed_heads:
            continue
        possible_heads, pos = [elem], 0
        while pos < len(possible_heads):
            head = possible_heads[pos]
            for child in head.children:
                if is_verb(child) and child.deprel in ["conj"]:
                    possible_heads.append(child)
            pos += 1
        processed_heads.update(possible_heads)
        possible_nodes = list(elem.children(following_only=True))
        following_children = elem.children(following_only=True)
        if len(following_children) > 0:
            while len(following_children) > 0:
                last_child = following_children.pop()
                if last_child.deprel not in ["obl"]:
                    continue
                curr_children = last_child.children(following_only=True)
                if len(curr_children) > 0:
                    for child in curr_children[::-1]:
                        if ((child.deprel != "conj" or child.feats.get("Case") == last_child.feats.get("Case"))
                                and all(x.deprel != "nsubj" for x in child.children)):
                            break
                        possible_nodes.append(child)
        while len(possible_nodes) > 0:
            node = possible_nodes.pop()
            try:
                if (node.deprel in ["conj"] and (is_noun(node) or is_adj(node)) and
                        all((child.deprel not in blocking_relations) for child in node.children) and
                        any(child.deprel == orphan_rel for child in node.children)):
                    for i in range(len(possible_heads)-1, -1, -1):
                        if possible_heads[i].ord < node.ord:
                            head_gap_pairs.append((possible_heads[i], node))
                            break
                    possible_nodes += list(node.children)
            except AttributeError:
                continue
    answer = []
    for head, gap in head_gap_pairs:
        first = [elem for elem in head.children if elem.deprel not in ["punct", "conj", "advcl"]
                 and is_possible_as_remnant(elem.form)]
        if is_copula(head):
            first.append(head)
        second = [elem for elem in gap.children() if elem.deprel == orphan_rel and
                  is_possible_as_remnant(elem.form) and (is_noun(elem) or is_adj(elem)) and
                  can_be_second_in_gap(elem, orphan_rel)]
        if len(second) > 0:
            second.append(gap)
            second.sort(key=(lambda x: x.ord))
            answer.append((head, first, second))
    return answer

DESCENDANTS_TO_EXCLUDE = ["punct", "cc", "orphan", "parataxis"]


def match_remnants(head_children: ListOfNodes, gap_children: ListOfNodes, head: Node):
    matched = [[] for _ in gap_children]
    for i, second in enumerate(gap_children):
        if second.deprel == "nsubj" or (i == 0 and second.deprel == "conj"):
            expected_deprel = "nsubj"
        else:
            expected_deprel = None
        for first in head_children:
            match = do_nodes_match(first, second)
            block_match = (first.ord - head.ord) * (2 * i - 1)
            if match > 0:
                matched[i].append((first, 2, match, first.form == second.form, block_match > 0))
            # if first.deprel not in ["nsubj", "obj", "obl", "nsubj:pass"]:
            #     continue
            for child in first.children:
                if not is_possible_as_remnant(child.form):
                    continue
                if child in gap_children:
                    break
                match = do_nodes_match(child, second)
                if match > 0:
                    matched[i].append((child, int(first.deprel == expected_deprel), match, child.form == second.form, block_match > 0))
    for stage in range(6):
        if stage == 0:
            filter_func = lambda x: True # единственный кандидат
        elif stage == 1:
            filter_func = lambda x: x[3]  # кандидат с той же словоформой
        elif stage == 2:
            filter_func = lambda x: x[4]
        elif stage == 3:
            filter_func = lambda x: (x[1] == 2 and x[2] > 1) # кандидат верхнего уровня
        elif stage == 4:
            filter_func = lambda x: (x[2] > 1)
            key_func = lambda x: (x[4], x[3], x[2], x[1])
        elif stage == 5:
            filter_func = lambda x: True
            key_func = lambda x: (x[4], x[3], x[2], x[1])
        if stage < 4:
            curr_matched = [list(filter(filter_func, x)) for x in matched]
        else:
            curr_matched = [[], []]
            for i, elem in enumerate(matched):
                filtered = list(filter(filter_func, elem))
                if len(filtered) > 1:
                    max_value = max(key_func(x) for x in filtered)
                    best_filtered = [x for x in filtered if key_func(x) == max_value]
                elif len(filtered) == 1:
                    best_filtered = filtered
                else:
                    continue
                curr_matched[i] = best_filtered
        for i, elem in enumerate(curr_matched):
            if len(elem) == 1:
                node = elem[0][0]
                curr_matched[1 - i] = [x for x in curr_matched[1-i] if
                                       (x[0] != node and (x[0] not in node.children or node == head))]
                if stage <= 2:
                    matched[1 - i] = [x for x in matched[1 - i] if
                                      (x[0] != node and (x[0] not in node.children or node == head))]
        if (len(curr_matched[0]) == 1 and len(curr_matched[1]) == 1
                and curr_matched[0][0][0] != curr_matched[1][0][0]):
            matched_nodes = [curr_matched[0][0][0], curr_matched[1][0][0]]
            for i, node in enumerate(matched_nodes):
                match_rel = gap_children[i].deprel
                if match_rel == "conj":
                    match_rel = "nsubj"
                matched_nodes[i] = normalize_node(node, match_rel=match_rel)
            return matched_nodes
                # matched[i] = [x for x in matched[i] if x[0] == node]
    return None



def extract_gaps(sent: Node, orphan_rel=None):
    gap_data = []
    orphan_rel = [] if orphan_rel is None else [orphan_rel] if isinstance(orphan_rel, str) else orphan_rel
    FUNCS = [orphan] + [partial(orphan, orphan_rel=x) for x in orphan_rel]
    for func in FUNCS:
        gap_data += func(sent.descendants)
    answer = []
    for head, head_children, gap_children in gap_data:
        if len(gap_children) != 2:
            continue
        gap, inverse = None, False
        head_remnants = match_remnants(head_children, gap_children, head)
        # head_remnants = []
        # for second in gap_children:
        #     matched = []
        #     for first in head_children:
        #         if are_equal_pos(first.upos, second.upos) and have_equal_case(first, second):
        #             matched.append(first)
        #         else:
        #             for child in first.children:
        #                 if are_equal_pos(child.upos, second.upos) and have_equal_case(child, second):
        #                     matched.append(child)
        #     if len(matched) == 1 and matched[0] not in head_remnants:
        #         head_remnants.append(matched[0])
        if head_remnants is not None:
            head = normalize_node(head)
            ord_to_exclude = [elem.ord for elem in gap_children]
            left_subtree_bounds = get_subtree_bounds(
                gap_children[0], deprel_to_exclude=DESCENDANTS_TO_EXCLUDE, ord_to_exclude=ord_to_exclude,
                deprel_checker=DEPREL_CHECKER)
            right_subtree_bounds = get_subtree_bounds(
                gap_children[1], deprel_to_exclude=DESCENDANTS_TO_EXCLUDE, ord_to_exclude=ord_to_exclude,
                deprel_checker=DEPREL_CHECKER)
            diff = right_subtree_bounds[0] - left_subtree_bounds[1]
            if diff < 0:
                diff = left_subtree_bounds[0] - right_subtree_bounds[1]
                head_remnants = head_remnants[::-1]
            if diff in [0, 1]:
                gap = right_subtree_bounds[0] if not inverse else left_subtree_bounds[0]
                # head_subtree = get_subtree_bounds(
                #     head, left_bound=head_remnants[0].ord, right_bound=head_remnants[1].ord)
                remnants_subtrees = [get_subtree_bounds(elem, ord_to_exclude=ord_to_exclude,
                                                        deprel_to_exclude=["punct", "cop"]) for elem in head_remnants]
                answer.append(((head.ord-1, head.ord), (gap, gap),
                               remnants_subtrees,
                               [left_subtree_bounds, right_subtree_bounds]))
    return gap_data, answer

def rearrange_answer(data):
    if len(data) == 0:
        return []
    head_key = [data[0][0]] + data[0][2]
    answer = [head_key]
    for elem in data:
        if elem[0] == data[0][0] and elem[2] == data[0][2]:
            answer.append([elem[1]] + elem[3])
    return np.array(answer).tolist()


CODES = ["TP", "error", "partial", "FN", "FP", "FP_gaps", "TN"]

def extract_return_code(corr, gaps, pred):
    if corr is None:
        code = "FP" if len(pred) > 0 else "FP_gaps" if len(gaps) > 0 else "TN"
    else:
        if sorted(pred) == sorted(corr):
            code = "TP"
        elif len(pred) > 0:
            code = "error"
        elif len(gaps) > 0:
            code = "partial"
        else:
            code = "FN"
    return code


if __name__ == "__main__":
    source_file, infile, model_name = "data/train.csv", "results/example_1.out", "orphan.nsubj.para.nmod.adj.appos-2000"
    (source_sents, labels), sents = read_data(source_file), read_parse_file(infile, max_sents=2000, parse=False)
    word_labels = []
    parsed_sents = [Conllu(filehandle=StringIO(sent)).read_tree() for sent in sents]
    for i, (source_sent, curr_labels, sent) in enumerate(zip(source_sents, labels, parsed_sents), 1):
        if curr_labels is not None:
            words = [elem.form for elem in sent.descendants]
            curr_labels = char_to_word_positions(source_sent, words, curr_labels)
        word_labels.append(curr_labels)
    gaps, answer = [], []
    for i, sent in enumerate(parsed_sents):
        if i % 500 == 0 and i > 0:
            print("{} sents parsed".format(i))
        curr_gaps, curr_answer = extract_gaps(sent, orphan_rel=["nsubj", "parataxis", "appos", "nmod"])
        curr_answer = rearrange_answer(curr_answer)
        gaps.append(curr_gaps)
        answer.append(curr_answer)
    indexes_by_codes = defaultdict(list)
    for i, (sent, corr_labels, curr_gaps, pred_labels) in enumerate(zip(source_sents, word_labels, gaps, answer)):
        code = extract_return_code(corr_labels, curr_gaps, pred_labels)
        indexes_by_codes[code].append(i)
    output_dir = "results/{}".format(model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for code in CODES:
        L = len(indexes_by_codes[code])
        print("{}\t{}\t{:.2f}".format(code, L, 100 * L / len(word_labels)))
        if code != "TN":
            outfile = os.path.join(output_dir, code + ".out")
            output_results(outfile, answer, gaps, word_labels, source_sents,
                           parsed_sents, indexes=indexes_by_codes[code],
                           output_trees=(code in ["FN", "error", "partial", "FP"]))

