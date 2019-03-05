import os
from collections import defaultdict

from common import *
from common import can_be_verb
from read_write import read_data, read_parse_file, read_gap_file
from gap_types import find_segment_root
from fix_tree import fix_tree, find_possible_right_subtree_heads

from udapi.core.node import Node, ListOfNodes
from udapi.core.document import Document
from udapi.block.write.textmodetrees import TextModeTrees


HYPHENS = "-—–"

def find_covering_edge(nodes, anchor):
    if anchor == 0 or anchor == len(nodes) - 1:
        return None, None
    index = anchor - 1
    while nodes[index].upos == "PUNCT":
        index -= 1
    curr_node: Node = nodes[index]
    prev_node: Node = None
    first, second = None, None
    while curr_node.ord > 0:
        if curr_node.ord > anchor:
            first, second = curr_node, prev_node
            break
        children = curr_node.children(following_only=True)
        if len(children) == 0 or children[-1].ord <= anchor + 1:
            prev_node = curr_node
            curr_node = curr_node.parent
            continue
        for child in curr_node.children(following_only=True):
            if child.ord > anchor + 1:
                first, second = curr_node, child
                break
        if first is not None:
            break
    if first is not None and not is_word(first.form):
        first, second = None, None
    if second is not None and not is_word(second.form):
        first, second = None, None
    return first, second


def find_verb_root(node, verb=None, from_known_gap=False, return_conj=True, promote_conj=True,
                   allow_nsubj=False, allow_a=True, allow_child=False):
    in_conj_branch, conj, prev = True, None, node
    if allow_nsubj and node.deprel in ["obl"] and can_be_verb(node.parent, allow_inf=False):
        verb = node.parent
    if verb is not None:
        conj = node
    initial_node = node
    if from_known_gap:
        initial_verb, verb, conj = verb, None, None
    # conj_is_known = False
    while verb is None and node.ord > 0:
        left_children = node.children(preceding_only=True)
        if can_be_verb(node, allow_inf=False) or node.form in ["можно", "нужно", "нельзя", "надо"]:
            verb = node
            break
        if from_known_gap and len(left_children) > 0 and left_children[0].form in HYPHENS and conj is None:
            conj = prev
            in_conj_branch = False
            if from_known_gap:
                break
        elif node.deprel in ["conj", "advcl"] and (conj is None or (in_conj_branch and have_equal_pos(node, conj))):
            conj = node
        elif node.deprel == "nmod" and node.children[0].form == ",":
            conj = node
        elif node.deprel == "appos" and node.children[0].form == ",":
            conj = node
        else:
            in_conj_branch = False
            if allow_a:
                children = node.children(preceding_only=True)
                if len(children) >= 2 and children[0].form + children[1].form == ",а":
                    conj = node
        if node.parent.ord == 0:
            break
        node, prev = node.parent, node
    if from_known_gap:
        verb = initial_verb
        if conj is None:
            conj = initial_node
    if promote_conj:
        children = verb.children(following_only=True) if verb is not None else node.children(following_only=True)
        for other in children:
            if other.ord >= prev.ord:
                break
            if other.deprel in ["conj", "advcl", "acl:relcl"] and is_verb(other, allow_part=False):
                verb = other
    if conj is None and allow_child:
        if initial_node.parent == verb:
            if initial_node.deprel in ["det", "amod", "parataxis"]:
                conj = initial_node
            if initial_node.deprel in ["obl", "nmod"]:
                if initial_node.children[0].form == ",":
                    conj = initial_node
                elif initial_node.ord > verb.ord:
                    for child in verb.children(following_only=True):
                        if child == initial_node:
                            break
                        if have_equal_case(child, initial_node, check_equal_pos=True):
                            conj = initial_node
        elif initial_node.deprel == "parataxis":
            conj = initial_node
    if verb is not None:
        verb = normalize_node(verb)
    return (verb, conj) if return_conj else verb


def promote_head(head, anchor):
    if head.ord < anchor:
        children = head.children(following_only=True)
        for other in children:
            if other.ord > anchor:
                break
            if other.deprel == "conj" and have_equal_pos(head, other):
                head = other
    return head


def find_orphan_edges(sent: ListOfNodes):
    answer = []
    for node in sent:
        if node.deprel == "orphan" and node.upos not in ["ADP", "PART"]:
            answer.append((node.parent, node))
    return answer

def find_a_edges(sent: ListOfNodes):
    answer = []
    for i, node in enumerate(sent[:-1]):
        if node.form != "," or sent[i+1].form not in ["а"] or len(sent[i+1].children) > 0:
            continue
        conj, parent = sent[i+1].form, node.parent
        children = parent.children
        if sent[i+1].parent != parent or children[0] != node:
            continue
        child = extract_possible_gap_children(parent, start=2)
        if child is not None:
            answer.append((parent, child))
    return answer

def extract_possible_gap_children(node, start=0):
    possible_children = []
    for child in node.children[start:]:
        if child.form in ["также"]:
            return None
        child_subtree = child.descendants
        if any(x.form in (list(HYPHENS) + ["также"]) for x in child_subtree):
            return None
        if child.deprel in ["conj", "acl:relcl", "parataxis"] and child.ord > node.ord:
            break
        if is_adj_noun_match(child, node, check_adj=True, check_noun=True):
            prep = extract_prep(node, return_form=False)
            if prep is None or prep.ord < child.ord:
                continue
        if child.deprel == "nmod" and not extract_prep(child) and child.feats.get("Case") == "Gen":
            continue
        if (child.deprel not in ["punct", "case", "flat:foreign", "mark",
                                 "cop", "flat:name", "nummod", "appos", "cc"] and
                child.upos != "ADP"):
            # if child.deprel not in ["punct", "case", "flat:foreign", "conj]:
            possible_children.append(child)
    nsubj = [child for child in possible_children
             if ("nsubj" in child.deprel and child.form not in ["это", "то"])]
    child = None
    if len(nsubj) == 1:
        child = nsubj[0]
    elif len(possible_children) == 1:
        if possible_children[0].deprel not in ["nsubj", "cc"]:
            child = possible_children[0]
    else:
        if len(possible_children) == 2:
            if possible_children[-1].deprel in ["nmod", "obl"]:
                if possible_children[0].deprel not in ["advmod", "nsubj", "cc"]:
                    child = possible_children[0]
            # if possible_children[0].form == "и":
            #     if possible_children[1].deprel not in ["advmod", "nsubj", "cc"]:
            #         child = possible_children[1]
    return child

def find_conj_nsubj_edges(sent: ListOfNodes):
    answer = []
    for node in sent:
        if can_be_verb(node, allow_inf=False) or node.feats.get("Variant") == "Short":
            possible_children = [(x, None) for x in node.children(following_only=True)]
            if node.deprel == "advcl" and node.feats.get("VerbForm") != "Conv":
                start, children = 0, node.parent.children
                while children[start] != node:
                    start += 1
                start += 1
                while start < len(children) and children[start].form in [",", "то"]:
                    start += 1
                if node.parent.form != "то":
                    possible_children.append((node.parent, start))
            for other, start in possible_children:
                if other.form in [":", "..."]:
                    break
                allow_inf = (node.feats.get("VerbForm") == "Inf")
                if can_be_verb(other, allow_inf=allow_inf):
                    break
                if other.feats.get("Variant") == "Short":
                    continue
                if other.upos == "ADV":
                    continue
                if start is not None or other.deprel in ["conj", "parataxis"]:
                    if start is None:
                        if (len(other.children) < 2 or other.children[0].form not in [","]):
                            continue
                        start = 1 + int(other.children[1].form.lower() == "a")
                    child = extract_possible_gap_children(other, start=start)
                    if child is not None:
                        answer.append((other, child, node))
            possible_heads = find_possible_right_subtree_heads(node)
            for head in possible_heads:
                if can_be_verb(head, allow_inf=False):
                    continue
                left_children = head.children(preceding_only=True)
                right_children = head.children(following_only=True)
                if len(left_children) < 1:
                    continue
                if left_children[0].form != ",":
                    continue
                for child in left_children[1:] + right_children:
                    if child.deprel == "mark" or child.form in ["где", "как", "откуда"] :
                        break
                    if child.deprel == "nsubj" and child.feats.get("Case") == "Nom":
                        answer.append((head, child, node))
                        break
                    # if child.deprel == "amod" and child.feats.get("Case") == "Nom":
                    #     answer.append((head, child, node))
                    #     break

    return sorted(set(answer), key=(lambda x: x[0].ord))


def is_nmod(node):
    return node.feats.get("Case") == "Gen" and extract_prep(node) is None


def has_inf(node):
    return node.upos == "ADV" and any(
        child.feats.get("VerbForm") == "Inf" and child.deprel == "csubj" for child in node.children)


def rearrange_remote_head(head, child, anchor):
    verb = None
    if head.ord > child.ord:
        return head, None
    head_children = head.children(following_only=True)
    if len(head_children) == 0:
        return head, None
    possible_head = None
    for i, head_child in enumerate(head_children):
        descendants = head_child.descendants(add_self=True)
        if len(descendants) > 1 and descendants[0].ord - 1 == anchor:
            possible_head = head if i == 0 else head_children[i-1]
        elif len(descendants) > 1 and descendants[-1].ord - 1 == anchor and head_child.deprel != "nmod":
            possible_head = head_child
        if possible_head is not None or descendants[0].ord > anchor:
            break
    if possible_head is None:
        return head, None
    if is_verb(head, allow_inf=False):
        verb = head
    while True:
        possible_head_children = possible_head.children
        if len(possible_head_children) == 0:
            break
        left, right = possible_head_children[0], possible_head_children[-1]
        if left.ord < possible_head.ord and left.form in ",аи" and possible_head.deprel != "advcl":
            head = possible_head
        elif right.ord > possible_head.ord and right.ord <= anchor:
            possible_head = right
            continue
        break
    if verb == head:
        verb = None
    return head, verb


def extract_gap_triples(root, verb_gap_pairs=None, return_gap=False):
    nodes = root.descendants
    answer, edges = [], set()
    if verb_gap_pairs is None:
        anchors = [node.ord - 1 for node in nodes if node.form in "-—–"]
        edges = set()
        for anchor in anchors:
            verb = None
            if anchor == 0 or anchor == len(nodes) - 1:
                continue
            # if nodes[anchor-1].form.isdigit() and nodes[anchor+1].form.isdigit():
            if nodes[anchor-1].upos == "NUM" and nodes[anchor+1].upos == "NUM":
                continue
            head, child = find_covering_edge(nodes, anchor)
            if head is not None:
                head, verb = rearrange_remote_head(head, child, anchor)
            if head is None or head.ord == 0 or is_verb(head, allow_inf=False, allow_part=False):
                continue
            if head.deprel == "parataxis" and child.feats.get("Case") == "Nom":
                continue
            if (child.upos in ["PART", "CCONJ", "SCONJ"] or child.deprel == "discourse" or
                    (child.upos == "ADP" and all(x.deprel != "fixed" for x in child.children))):
                continue
            if head.ord < child.ord:
                head = promote_head(head, anchor=anchor)
            edges.add((head, child, verb, True, False, anchor))
        for head, child in find_orphan_edges(nodes):
            if head.ord < child.ord:
                head = promote_head(head, anchor=child.ord)
            edges.add((head, child, None, True, False, None))
        for head, child in find_a_edges(nodes):
            edges.add((head, child, None, True, False, None))
        for head, child, node in find_conj_nsubj_edges(nodes):
            edges.add((head, child, node, False, False, None))
    else:
        for verb, anchor in verb_gap_pairs:
            gap_anchor = anchor
            if nodes[anchor-1].form in HYPHENS:
                anchor -= 1
            # else:
            #     right_descendants = nodes[anchor].descendants(following_only=True)
            #     if len(right_descendants) > 0 and right_descendants[-1].ord >= len(nodes) - 1:
            #         anchor -= 1
            head, child = find_covering_edge(nodes, anchor)
            if head is not None:
                head, _ = rearrange_remote_head(head, child, anchor)
            if head is None or head.upos in ["PUNCT", "ADP"] or child.upos in ["PUNCT", "ADP"]:
                continue
            if head.ord < child.ord:
                head = promote_head(head, anchor=anchor)
            edges.add((head, child, nodes[verb], True, True, gap_anchor))
    for head, child, verb, allow_child, from_known_gap, gap in sorted(edges, key=lambda x: x[0].form):
        verb, head = find_verb_root(head, verb=verb, from_known_gap=from_known_gap,
                                    allow_nsubj=False, allow_child=allow_child)
        if verb is not None and head is not None:
            if child.deprel == "advmod":
                if all(other.deprel != "advmod" for other in verb.children):
                    continue
            to_append = [verb.ord-1] + sorted([head.ord-1, child.ord-1])
            if return_gap:
                to_append.append(gap)
            answer.append(tuple(to_append))
    answer = sorted(set(answer))
    return answer


def get_extraction_key(corr, pred):
    if corr is None:
        return {"TN" if len(pred) == 0 else "FP"}
    answer = set()
    for elem in corr:
        if elem == "error":
            answer.add("error")
        elif elem in pred:
            answer.add("TP")
        elif any(len([x for x in elem if x in other]) >= 2 for other in pred):
            answer.add("wrong")
        else:
            answer.add("FN")
    for elem in pred:
        if elem not in corr and "error" not in corr:
            if any(len([x for x in elem if x in other]) >= 2 for other in corr):
                answer.add("wrong")
            else:
                answer.add("FP")
            break
    return answer


def output_results(outfile, answer, corr_answer, sents, parsed_sents,
                   indexes=None, output_trees=False, output_correct=False):
    if indexes is None:
        indexes = list(range(len(corr_answer)))
    with open(outfile, "w", encoding="utf8") as fout:
        for i in indexes:
            writer = TextModeTrees(filehandle=fout, attributes="form,upos,deprel,ord,feats")
            sent = parsed_sents[i].descendants
            fout.write("{}\t{}\n{}\n".format(i, sents[i], "=" * 40))
            if output_trees:
                writer.before_process_document(Document())
                writer.process_tree(parsed_sents[i])
                writer.after_process_document(Document())
                fout.write("=" * 40 + "\n")
            if len(answer[i]) > 0:
                for elem in answer[i]:
                    nodes = [sent[j] for j in elem]
                    fout.write("\t".join("{}-{}".format(node.ord, node.form) for node in nodes) + "\n")
            if output_correct and corr_answer[i] is not None:
                fout.write("=" * 15 + " CORRECT " + "=" * 15 + "\n")
                for elem in corr_answer[i]:
                    if elem == "error":
                        fout.write("error\n")
                    else:
                        nodes = [sent[j] for j in elem]
                        fout.write("\t".join("{}-{}".format(node.ord, node.form) for node in nodes) + "\n")
            fout.write("\n")

FROM_KNOWN_GAPS = False
gap_file = "results/verb_gap_pairs.out"

if __name__ == "__main__":
    tree_fixes = ["fix_right_nsubj", "fix_adv_adj"]
    source_file, infile = "data/dev.csv", "results/example_dev.out"
    output_dir, model_name = "gap_results_dev", "predicted"
    (source_sents, labels), parsed_sents = read_data(source_file), read_parse_file(infile, parse=False)
    word_labels, gap_labels = [], []
    # parsed_sents = [Conllu(filehandle=StringIO(sent)).read_tree() for sent in sents]
    for i, (source_sent, curr_labels, sent) in enumerate(zip(source_sents, labels, parsed_sents), 1):
        curr_gap_labels = []
        if curr_labels is not None:
            words = [elem.form for elem in sent.descendants]
            curr_word_labels = char_to_word_positions(source_sent, words, curr_labels)
            curr_labels = []
            for gap_block, left_block, right_block in curr_word_labels[1:]:
                left_subtree_head = find_segment_root(sent, *left_block)
                right_subtree_head = find_segment_root(sent, *right_block)
                root_error_code = 2 * int(left_subtree_head is None) + int(right_subtree_head is None)
                if root_error_code > 0:
                    curr_labels.append("error")
                else:
                    curr_labels.append((curr_word_labels[0][0][0], left_subtree_head.ord-1, right_subtree_head.ord-1))
                curr_gap_labels.append(gap_block[0])
        word_labels.append(curr_labels)
        gap_labels.append(curr_gap_labels)
    answer = []
    if FROM_KNOWN_GAPS:
        gap_data = list(zip(word_labels, gap_labels))
    elif gap_file is not None:
        gap_indexes = read_gap_file(gap_file, len(parsed_sents))
        for elem in gap_indexes:
            elem[0] = [(x, None, None) for x in elem[0]]
    else:
        gap_data = None
    for i, sent in enumerate(parsed_sents):
        if len(tree_fixes) > 0:
            curr_fixes = fix_tree(sent, tree_fixes)
        if gap_data is not None:
            curr_gap_data = gap_data[i]
            if curr_gap_data[0] is None or "error" in curr_gap_data[0]:
                answer.append([])
                continue
            verb_gap_pairs = [(first[0], second) for first, second in zip(*curr_gap_data)]
        else:
            verb_gap_pairs = None
        answer.append(extract_gap_triples(sent, verb_gap_pairs=verb_gap_pairs))
    stats = defaultdict(list)
    output_dir = "{}/{}".format(output_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for i, (corr, pred) in enumerate(zip(word_labels, answer)):
        keys = get_extraction_key(corr, pred)
        for key in keys:
            stats[key].append(i)
    for key in ["TP", "wrong", "FP", "FN", "TN", "error"]:
        print(key, len(stats[key]))
        outfile = os.path.join(output_dir, key)
        output_results(outfile, answer, word_labels, source_sents,
                       parsed_sents, stats[key], output_trees=(key[0] != "T"),
                       output_correct=(key in ["FN", "wrong"]))
    fn_indexes = stats["FN"]
    for start in range(0, len(fn_indexes), 10):
        print(fn_indexes[start:start+10])
