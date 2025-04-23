# hiearchy.py  (in the same folder)
LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def build_paths(categories: list[str]) -> dict[str, list[str]]:
    from fathomnet.api.worms import get_ancestors

    def extract_lineage(node):
        lineage = {}
        curr = node
        while curr:
            lineage[curr.rank.lower()] = curr.scientific_name
            curr = getattr(curr, "children", [None])[0]
        return lineage

    paths = {}
    for cat in categories:
        try:
            tree = get_ancestors(cat)
            ranks = extract_lineage(tree)
        except:
            ranks = {}
        paths[cat] = [ranks.get(lvl, "UNK") for lvl in LEVELS]
    return paths
