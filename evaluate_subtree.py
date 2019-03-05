import os
from collections import defaultdict
from udapi.core.document import Document
from udapi.block.write.textmodetrees import TextModeTrees

from subtree import find_gap_subtree_spans, to_subtree_heads
from read_write import read_data, read_parse_file
from fix_tree import fix_tree

from match_gaps import char_to_word_labels, get_subtree_extraction_key, find_gap_position


def output_gap_stats(sents, indexes, outfile):
    with open(outfile, "w", encoding="utf8") as fout:
        writer = TextModeTrees(filehandle=fout, attributes="form,upos,deprel,ord,feats")
        for index, gap_index, corr_gap, pred_gap in indexes:
            curr_sent = sents[index].descendants
            writer.before_process_document(Document())
            writer.process_tree(parsed_sents[index])
            writer.after_process_document(Document())
            fout.write("{}\t|\t{}\n\n".format(curr_sent[pred_gap].form, curr_sent[corr_gap].form))

if __name__ == "__main__":
    tree_fixes = ["fix_adv_adj"]
    source_file, infile = "data/train.csv", "results/example_1.out"
    output_dir, model_name = "span_results_4000", "basic.hyphen"
    gap_outfile = os.path.join(output_dir, "wrong_gaps.out")
    (source_sents, labels), parsed_sents = read_data(source_file), read_parse_file(infile, max_sents=4000, parse=False)
    word_sents = [[elem.form for elem in sent.descendants] for sent in parsed_sents]
    word_labels = char_to_word_labels(source_sents, word_sents, labels)
    subtree_labels = to_subtree_heads(parsed_sents, word_labels)
    subtree_spans = []
    gap_stats = [[], []]
    for i, (sent, curr_labels) in enumerate(zip(parsed_sents, subtree_labels)):
        if len(tree_fixes) > 0:
            curr_fixes = fix_tree(sent, tree_fixes)
        if curr_labels is None:
            subtree_spans.append([])
            continue
        curr_subtree_spans = []
        verb = curr_labels[0][0]
        for r, (gap, left, right) in enumerate(curr_labels[1:], 1):
            if r < len(curr_labels) - 1:
                rightmost = curr_labels[r+1][1]
            else:
                rightmost = None
            if left is not None and right is not None:
                pred_gap = find_gap_position(sent, left, right)
                gap_stats[int(pred_gap != gap)].append((i, r, gap, pred_gap))
                spans = find_gap_subtree_spans(sent, verb, left, right, rightmost)
            else:
                spans = None
            curr_subtree_spans.append(spans)
        subtree_spans.append(curr_subtree_spans)
    print("Correct gaps: {}, wrong: {}".format(*[len(x) for x in gap_stats]))
    output_gap_stats(parsed_sents, gap_stats[1], gap_outfile)
    stats = defaultdict(list)
    for i, (corr, pred) in enumerate(zip(word_labels, subtree_spans)):
        if corr is None:
            continue
        for j, elem in enumerate(pred):
            key = get_subtree_extraction_key(corr[j+1], pred[j])
            stats[key].append((i, j))
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    # KEYS = ["TP", "partial", "wrong", "FN"]
    # for key in KEYS:
    #     print(key, len(stats[key]))
    #     outfile = os.path.join(output_dir, key)
    #     output_matching_results(outfile, subtree_labels, subtree_labels,
    #                             subtree_spans, word_labels,
    #                             source_sents, parsed_sents, stats[key],
    #                             output_trees=(key != "TP"),
    #                             output_correct=(key != "TP"),
    #                             compare_with_first=False)