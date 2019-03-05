from common import *
from udapi.core.feats import Feats

FIX_FUNCS = ["fix_last_dep", "fix_right_nsubj", "fix_adv_adj"]

def fix_tree(root, fix_funcs=None):
    if fix_funcs is None:
        fix_funcs = FIX_FUNCS
    fixes = []
    for func in fix_funcs:
        if isinstance(func, str):
            func = eval(func)
        is_fixed = func(root)
        if is_fixed:
            fixes.append(func.__name__)
    return fixes


def fix_right_nsubj(nodes):
    is_fixed = False
    if isinstance(nodes, Node):
        nodes = nodes.descendants(add_self=True)
    for node in nodes:
        if node.deprel == "nsubj" and node.feats.get("Case") == "Nom":
            if node.parent.ord == 0 or node.parent.ord > node.ord:
                continue
            if not is_verb(node.parent, allow_short=False, allow_part=False, allow_inf=False):
                continue
            if not is_verb_noun_match(node, node.parent):
                node.deprel = "conj"
                is_fixed = True
    return is_fixed


def fix_adv_adj(nodes):
    is_fixed = False
    if isinstance(nodes, Node):
        nodes = nodes.descendants(add_self=True)
    for node in nodes:
        if is_short(node) and node.feats["Gender"] == "Neut":
            if node.parent.ord == 0 or node.parent.ord > node.ord:
                continue
            if is_verb(node.parent, allow_inf=False, allow_short=False, allow_part=True):
                has_nsubj_child = False
                for child in node.children:
                    if child.deprel == "nsubj" and child.feats["Gender"] == "Neut":
                        has_nsubj_child = True
                    if child.feats["VerbForm"] == "Inf":
                        has_nsubj_child = True
                if not has_nsubj_child:
                    node.upos = "ADV"
                    node.feats = Feats(dict())
                    node.deprel = "advmod"
                    is_fixed = True
    return is_fixed

def fix_last_dep(nodes):
    is_fixed = False
    if isinstance(nodes, Node):
        nodes = nodes.descendants(add_self=True)
    verbs = [(node, node) for node in nodes if can_be_verb(node)]
    possible_heads = []
    while len(verbs) > 0:
        curr, verb = verbs.pop()
        has_added = False
        for child in curr.children(following_only=True):
            if child.deprel in ["conj", "punct"] and not has_added:
                continue
            elif child.deprel in ["obl", "nmod", "obj"]:
                possible_heads.append((child, verb))
                verbs.append((child, verb))
                has_added = True
            else:
                break
    for head, verb in possible_heads:
        for child in head.children(following_only=True)[::-1]:
            if (child.deprel in ["conj"] and child.feats.get("Case") == "Nom" and
                    child.form.isalpha() and not have_equal_case(child, head)):
                for other in head.children(following_only=True)[::-1]:
                    if other == child:
                        break
                    other.parent = verb
                child.parent = verb
                child.deprel = "conj"
                child.misc = "fix: conj, from {} to {}".format(head.ord, verb.ord)
                is_fixed = True
                break
    return is_fixed


def find_possible_right_subtree_heads(verb):
    subtree_heads = [(verb, verb)]
    possible_heads, answer = [], []
    while len(subtree_heads) > 0:
        curr, verb = subtree_heads.pop()
        has_added = False
        found_child = None
        is_verb_short = is_short(verb)
        for child in curr.children(following_only=True):
            if child.deprel in ["punct", "xcomp"] and not has_added:
                continue
            elif child.deprel in ["obl", "nmod", "obj", "advmod", "nsubj"] and not is_copula(child, only_adj=False):
                found_child = child
            elif child.deprel == "conj":
                if not is_verb(child, allow_inf=False, allow_short=is_verb_short):
                    found_child = child
                if is_short(child) and child.feats.get("Gender") == "Neut":
                    found_child = child
            if found_child is not None:
                if found_child.form != "как":
                    subtree_heads.append((found_child, verb))
                    possible_heads.append(found_child)
                    has_added = True
            if (child.deprel  in "conj" and not has_added) or child.deprel == "nsubj":
                continue
            break
    for head in possible_heads:
        if head.feats.get("Case") != "Nom" and head.deprel != "nsubj":
            answer.append(head)
        for child in head.children(following_only=True)[::-1]:
            if (child.deprel in ["conj"] and child.form.isalpha()):
                answer.append(child)
                break
    return answer


