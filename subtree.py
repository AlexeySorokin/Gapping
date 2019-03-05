import bisect

from common import *
from gap_types import find_segment_root


def get_subtree_bounds(root, left_bound=None, right_bound=None,
                       deprel_to_exclude=None, sent=None,
                       exclude_conj=False, ord_to_exclude=None, deprel_checker=None):
    if deprel_to_exclude is None:
        deprel_to_exclude = []
    elif isinstance(deprel_to_exclude, str):
        deprel_to_exclude = [deprel_to_exclude]
    if deprel_checker is None:
        deprel_checker = lambda x: True
    if ord_to_exclude is None:
        ord_to_exclude = []
    elif isinstance(ord_to_exclude, int):
        ord_to_exclude = [ord_to_exclude]
    ord_to_exclude.sort()
    left_nodes, right_nodes = root.descendants(preceding_only=True), root.descendants(following_only=True)
    for i, node in enumerate(left_nodes[::-1]):
        if node.ord != root.ord - i - 1:
            left_nodes = left_nodes[len(left_nodes)-i:]
            break
    for i, node in enumerate(right_nodes):
        if node.ord != root.ord + i + 1:
            right_nodes = right_nodes[:i]
            break
    nodes = left_nodes + [root] + right_nodes
    # nodes = root.descendants(add_self=True)
    offset = nodes[0].ord
    if left_bound is None:
        left_bound = nodes[0].ord - 1
    if right_bound is None:
        right_bound = nodes[-1].ord + 1
    if len(ord_to_exclude) > 0:
        index = bisect.bisect_left(ord_to_exclude, root.ord)
        if index > 0:
            left_bound = max(left_bound, ord_to_exclude[index-1])
        while index < len(ord_to_exclude) and ord_to_exclude[index] == root.ord:
            index += 1
        if index < len(ord_to_exclude):
            right_bound = min(right_bound, ord_to_exclude[index])
    for child in root.children(preceding_only=True)[::-1]:
        is_deprel_ok = (child.deprel not in deprel_to_exclude or child.form in SPECIAL_WORDS)
        is_deprel_ok &= deprel_checker(child)
        if not is_deprel_ok:
            left_bound = max(left_bound, child.descendants(add_self=True)[-1].ord)
        if child.ord <= left_bound:
            break
    right_children = root.children(following_only=True)
    for i, child in enumerate(right_children):
        is_deprel_ok = (child.deprel not in deprel_to_exclude or child.form in SPECIAL_WORDS)
        is_deprel_ok &= deprel_checker(child)
        child_descendants = child.descendants(add_self=True)
        if child.deprel in ["nmod", "obl"] and child.children(add_self=True)[0].form == ",":
            is_deprel_ok = False
        if child.deprel in ["conj"] and exclude_conj:
            if is_verb(child, allow_inf=False, allow_short=False, allow_part=False):
                is_deprel_ok = False
        if not is_deprel_ok:
            right_bound = min(right_bound, child_descendants[0].ord)
        if child.ord >= right_bound:
            break
    if left_bound >= offset:
        left_bound = nodes[left_bound-offset]
        while True:
            parent = left_bound.parent
            if parent.ord < left_bound.ord and parent.children[-1] != left_bound:
                left_bound = parent
            else:
                break
        left_bound = left_bound.descendants(add_self=True)[-1]
        left = left_bound.ord
    else:
        left = offset - 1
    if right_bound <= nodes[-1].ord:
        right_bound = nodes[right_bound-offset]
        while True:
            parent = right_bound.parent
            if parent.ord > right_bound.ord and parent.children[0] != right_bound:
                right_bound = parent
            else:
                break
        right_bound = right_bound.descendants(add_self=True)[0]
        right = right_bound.ord - 1
    else:
        right = nodes[-1].ord
    if right - offset < len(nodes) and nodes[right-offset].form in ".-,–-—":
        node = nodes[right - offset]
        if node.form != "." or node.parent.form not in ["г", "млн"]:
            right -= 1
    return left, right


def find_gap_subtree_spans(sent, verb, left, right, rightmost=None):
    sent = sent.descendants
    verb, left, right = sent[verb], sent[left], sent[right]
    ord_to_exclude = [verb.ord, left.ord, right.ord]
    if rightmost is not None:
        ord_to_exclude.append(rightmost+1)
    left_subtree_span = get_subtree_bounds(
        left, ord_to_exclude=ord_to_exclude, deprel_to_exclude=["punct", "cc", "orphan", "mark"],
        sent=sent, exclude_conj=True)
    right_subtree_span = get_subtree_bounds(
        right, ord_to_exclude=ord_to_exclude, deprel_to_exclude=["punct", "cc", "orphan", "mark"],
        sent=sent, exclude_conj=True)
    return left_subtree_span, right_subtree_span


def find_subtree_spans(sent, verb, left_head, right_head, bound=None):
    sent = sent.descendants
    verb, left_head, right_head = sent[verb], sent[left_head], sent[right_head]
    ord_to_exclude = [right_head.ord] if bound is None else [right_head.ord, bound+1]
    left_subtree_span = get_subtree_bounds(
        left_head, ord_to_exclude=ord_to_exclude,
        deprel_to_exclude=["punct", "cop"])
    right_subtree_span = get_subtree_bounds(
        right_head, ord_to_exclude=ord_to_exclude,
        deprel_to_exclude=["punct", "cop"])
    return left_subtree_span, right_subtree_span


def to_subtree_heads(sents, labels):
    answer = []
    for sent, curr_labels in zip(sents, labels):
        if curr_labels is None:
            curr_answer = None
        else:
            curr_answer = []
            for elem in curr_labels:
                curr_elem = [elem[0][0]] + [None] * (len(elem) - 1)
                for i, block in enumerate(elem[1:], 1):
                    if block is not None:
                        head = find_segment_root(sent, *block)
                        if head is not None:
                            head = head.ord - 1
                        curr_elem[i] = head
                curr_answer.append(curr_elem)
        answer.append(curr_answer)
    return answer