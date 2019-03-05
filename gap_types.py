import os
from io import StringIO
from collections import defaultdict
from functools import partial

from common import *
#from common import do_nodes_match
from read_write import read_data, read_parse_file, output_results, make_phrase_repr

from udapi.block.read.conllu import Conllu
from udapi.core.node import Node, ListOfNodes, find_minimal_common_treelet
from udapi.core.document import Document
from udapi.block.write.textmodetrees import TextModeTrees


def find_segment_root(tree: ListOfNodes, start, end, only_in_segment=False):
    if isinstance(tree, Node):
        tree = tree.descendants
    if start < 0:
        return None
    last_in_segment = set()
    for node in tree[start:end]:
        while node.ord != 0:
            parent = node.parent
            if parent.ord <= start or parent.ord > end:
                last_in_segment.add(node.ord-1)
                break
            node = parent
    if len(last_in_segment) == 1:
        answer = last_in_segment.pop()
        is_in_segment = (answer >= start and answer < end)
        if not only_in_segment or is_in_segment:
            return tree[answer]
        else:
            return None
    else:
        return None


def make_branch(node: Node):
    answer = [node]
    while node.parent.ord:
        node = node.parent
        answer.append(node)
    answer = answer[::-1]
    return answer


def find_gap_tree_type(head_block, left_gap, right_gap):
    if head_block[1] != head_block[0] + 1:
        return None, None, None, None, "long head"
    if left_gap[0] < 0:
        return None, None, None, None, "no left gap dependant"
    if right_gap[0] < 0:
        return None, None, None, None, "no right gap dependant"
    head = sent.descendants[head_block[0]]
    left_gap_root = find_segment_root(sent, *left_gap)
    right_gap_root = find_segment_root(sent, *right_gap)
    root_error_code = 2 * int(left_gap_root is None) + int(right_gap_root is None)
    if root_error_code > 0:
        messages = {1: "no right gap subtree head", 2: "no left gap subtree head", 3: "no gap subtree heads"}
        return None, None, None, None, messages[root_error_code]
    nodes = [head, left_gap_root, right_gap_root]
    branches = [make_branch(node) for node in nodes]
    head_depth, left_depth, right_depth = [len(elem) for elem in branches]
    min_gap_depth = min(left_depth, right_depth)
    # if head_depth >= min_gap_depth:
    #     return None, None, None, "no head domination"
    if left_depth == right_depth:
        return None, None, None, None, "no gap domination"
    if left_depth < right_depth:
        gap_type, gap_upper, gap_lower = "left", left_gap_root, right_gap_root
        longest_branch = branches[2]
    else:
        gap_type, gap_lower, gap_upper = "right", left_gap_root, right_gap_root
        longest_branch = branches[1]
    if longest_branch[min_gap_depth-1] != gap_upper:
        return None, None, None, None, "no gap domination"
    if head_depth >= len(longest_branch) or longest_branch[head_depth-1] != head:
        for common_depth, (first, second) in enumerate(zip(branches[0], longest_branch)):
            if first != second:
                break
        head_path = [(elem.upos, elem.deprel) for elem in longest_branch[common_depth:head_depth]]
    else:
        head_path, common_depth = [], head_depth
        # return None, None, None, "no head domination"
    first_path = [(elem.upos, elem.deprel) for elem in longest_branch[common_depth:min_gap_depth]]
    second_path = [(elem.upos, elem.deprel) for elem in longest_branch[min_gap_depth:]]
    return gap_type, head_path, first_path, second_path, "ok"


if __name__ == "__main__":
    source_file, infile = "data/train.csv", "results/example_1.out"
    outdir = "results/tree_stats_2000"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    (source_sents, labels), sents = read_data(source_file), read_parse_file(infile, parse=False)
    word_labels = []
    parsed_sents = [Conllu(filehandle=StringIO(sent)).read_tree() for sent in sents]
    for i, (source_sent, curr_labels, sent) in enumerate(zip(source_sents, labels, parsed_sents), 1):
        if curr_labels is not None:
            words = [elem.form for elem in sent.descendants]
            curr_labels = char_to_word_positions(source_sent, words, curr_labels)
        word_labels.append(curr_labels)
    stats = defaultdict(list)
    for i, (curr_labels, sent) in enumerate(zip(word_labels, parsed_sents)):
        if curr_labels is None:
            continue
        if i >= 2000:
            break
        head_block = curr_labels[0][0]
        for elem in curr_labels[1:]:
            _, left_gap, right_gap = elem
            gap_data = (head_block, left_gap, right_gap)
            to_append = (i, source_sents[i], gap_data, sent)
            gap_type, head_path, first_path, second_path, error_type =\
                find_gap_tree_type(head_block, left_gap, right_gap)
            if error_type == "ok":
                key = (gap_type, "-".join("_".join(elem) for elem in head_path) if len(head_path) > 0 else "HEAD",
                       "-".join("_".join(elem) for elem in first_path),
                       "-".join("_".join(elem) for elem in second_path))
            else:
                key = (error_type,)
            stats[key].append(to_append)
    stats = sorted(stats.items(), key=(lambda x: (len(x[1]))), reverse=True)
    with open(os.path.join(outdir, "counts.out"), "w", encoding="utf8") as fout:
        for j, (key, key_stats) in enumerate(stats):
            fout.write("{}\t{}\n".format(" ".join(key), len(key_stats)))
    for j, (key, key_stats) in enumerate(stats):
        if j < 20:
            print(" ".join(key), len(key_stats))
        with open(os.path.join(outdir, " ".join(key) + ".out"), "w", encoding="utf8") as fout:
            writer = TextModeTrees(filehandle=fout, attributes="form,upos,deprel,ord,feats")
            for i, text, gap_data, parse in key_stats:
                fout.write("{}\t{}\n".format(i, text))
                fout.write("-" * 40 + "\n")
                for start, end in gap_data:
                    fout.write(make_phrase_repr(parse, start, end) + "\n")
                fout.write("-" * 40 + "\n")
                writer.before_process_document(Document())
                writer.process_tree(parsed_sents[i])
                writer.after_process_document(Document())
        # head = sent.descendants[head_block[0]]
        # left_gap_root = find_segment_root(sent, *left_gap)
        # right_gap_root = find_segment_root(sent, *right_gap)
        #
        # print(head.ord, head.form)
        # print(gap_head.ord, gap_head.form)
        # print(left_gap_root.ord, left_gap_root.form)
        # print(right_gap_root.ord, right_gap_root.form)