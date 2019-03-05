from collections import  defaultdict

from read_write import read_data, read_parse_file
from fix_tree import fix_tree

from read_write import get_start_positions, read_gap_file
from find_gap import extract_gap_triples
from match_gaps import char_to_word_labels, find_remnant, find_gap_position
from subtree import *


def make_char_answer(sent, word_starts, verb, remnants, gap, gap_remnants):
    answer = []
    answer.append((word_starts[verb], word_starts[verb]+len(sent[verb])))
    for left, right in remnants:
        answer.append((word_starts[left], word_starts[right-1]+len(sent[right-1])))
    answer.append((word_starts[gap], word_starts[gap]))
    for left, right in gap_remnants:
        answer.append((word_starts[left], word_starts[right-1]+len(sent[right-1])))
    return answer

def write_output(sents, answer, outfile):
    with open(outfile, "w", encoding="utf8") as fout:
        fout.write("\t".join(["text", "class", "cV", "cR1", "cR2", "V", "R1", "R2"]) + "\n")
        for sent, curr_answer in zip(sents, answer):
            if TO_RESOLVE:
                label = int(len(curr_answer) > 0)
            else:
                label = curr_answer
            fout.write("{}\t{}".format(sent, label))
            if TO_RESOLVE and len(curr_answer) > 0:
                fout.write("\t" + "\t".join("{}:{}".format(*x) for x in curr_answer[0][:3]))
                for i in range(3, 6):
                    fout.write("\t" + " ".join("{}:{}".format(*elem[i]) for elem in curr_answer))
            fout.write("\n")


FROM_GOLD = False


gap_file = "results/verb_gap_pairs.out"
TO_RESOLVE = True


def make_subtree_spans(sent, word_sent, word_starts, data):
    """
    data = [(verb, left_rem, right_rem, gap, left, right), ...]

    :param sent:
    :param data:
    :return:
    """
    answer = []
    right_bound_for_verb = None
    right_bounds = [elem[4] for elem in data if elem[4] is not None]
    if len(right_bounds) > 0:
        right_bound_for_verb = right_bounds[0]
    for r, (verb, left_head, right_head, gap, left, right) in enumerate(data):
        if left is None or right is None:
            continue
        if r < len(data):
            right_bounds = [elem[4] for elem in data[r + 1:] if elem[4] is not None]
            right_bound = right_bounds[0] if len(right_bounds) > 0 else None
        else:
            right_bound = None
        curr_subtree_spans = find_gap_subtree_spans(sent, verb, left, right, rightmost=right_bound)
        if left_head is None:
            left_head, right_head = find_remnant(sent, verb, left, right, break_ties=True)
        if left_head is not None and right_head is not None:
            curr_remnant_spans = find_subtree_spans(sent, verb, left_head, right_head, bound=right_bound_for_verb)
            curr_char_positions = make_char_answer(word_sent, word_starts, verb,
                                                   curr_remnant_spans, gap, curr_subtree_spans)
            answer.append(curr_char_positions)
    return answer



if __name__ == "__main__":
    tree_fixes = ["fix_adv_adj"]
    source_file, infile, outfile = "data/dev.csv", "results/example_dev.out", "runs/gap_dev_short.csv"
    # source_file, infile, outfile = "data/test.csv", "results/test.out", "runs/gap_test_neural.csv"
    (source_sents, labels), parsed_sents = read_data(source_file, max_sents=100), read_parse_file(infile, parse=False, max_sents=100)
    word_sents = [[elem.form for elem in sent.descendants] for sent in parsed_sents]
    word_starts = []
    for i, (sent, words) in enumerate(zip(source_sents, word_sents)):
        word_starts.append(get_start_positions(sent, words))
    if FROM_GOLD:
        word_labels = char_to_word_labels(source_sents, word_sents, labels)
        subtree_labels = to_subtree_heads(parsed_sents, word_labels)
    if gap_file is not None:
        verb_gap_data = read_gap_file(gap_file, len(parsed_sents))
    else:
        verb_gap_data = None
    answer = []
    for i, (sent) in enumerate(parsed_sents):
        if len(tree_fixes) > 0:
            curr_fixes = fix_tree(sent, tree_fixes)
        curr_answer = []
        if FROM_GOLD:
            curr_labels = subtree_labels[i]
            if curr_labels is None:
                answer.append([])
                continue
            verb = curr_labels[0][0]
            gap_triples = curr_labels[1:]
        else:
            if verb_gap_data is not None:
                curr_gap_data = list(zip(*verb_gap_data[i]))
            else:
                curr_gap_data = None
            if not TO_RESOLVE:
                answer.append(int(len(curr_gap_data) > 0))
                continue
            triples = extract_gap_triples(sent, verb_gap_pairs=curr_gap_data)
            gap_triples = []
            if len(triples) > 0:
                verbs = defaultdict(int)
                for triple in triples:
                    verbs[triple[0]] += 1
                verb = max(verbs.keys(), key=(lambda x: verbs[x]))
                pairs = [elem[1:] for elem in triples if elem[0] == verb]
                for pair in pairs:
                    gap = find_gap_position(sent, *pair)
                    if gap is not None:
                        gap_triples.append((gap, *pair))
            if len(gap_triples) == 0:
                answer.append([])
                continue

        curr_full_gap_data = []
        for elem in gap_triples:
            curr_full_gap_data.append((verb, None, None, *elem))
        curr_answer = make_subtree_spans(sent, word_sents[i], word_starts[i], curr_full_gap_data)
        answer.append(curr_answer)
        # right_bound_for_verb = None
        # right_bounds = [elem[1] for elem in gap_triples if elem[1] is not None]
        # if len(right_bounds) > 0:
        #     right_bound_for_verb = right_bounds[0]
        # for r, (gap, left, right) in enumerate(gap_triples):
        #     if left is None or right is None:
        #         continue
        #     if r < len(gap_triples):
        #         right_bounds = [elem[1] for elem in gap_triples[r+1:] if elem[1] is not None]
        #         right_bound = right_bounds[0] if len(right_bounds) > 0 else None
        #     else:
        #         right_bound = None
        #     curr_subtree_spans = find_gap_subtree_spans(sent, verb, left, right, rightmost=right_bound)
        #     left_head, right_head = find_remnant(sent, verb, left, right, break_ties=True)
        #     if left_head is not None and right_head is not None:
        #         curr_remnant_spans = find_subtree_spans(sent, verb, left_head, right_head, bound=right_bound_for_verb)
        #         curr_char_positions = make_char_answer(word_sents[i], word_starts[i], verb,
        #                                                curr_remnant_spans, gap, curr_subtree_spans)
        #         curr_answer.append(curr_char_positions)
        # answer.append(curr_answer)
    write_output(source_sents, answer, outfile)
    if FROM_GOLD:
        pass


