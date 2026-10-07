from graphlib import TopologicalSorter


def build_plan_graph(
    need_notes: bool,
    need_style: bool,
    need_profile: bool,
    need_glossary: bool,
):
    graph = {}

    if need_notes:
        graph["notes"] = set()

    if need_style:
        graph["style"] = set()

    if need_profile:
        graph["profile"] = set()

    if need_glossary:
        graph["glossary"] = set()

    # Reply must wait for all independent tool branches.
    graph["reply"] = set(graph.keys())

    return graph


def get_execution_order(
    need_notes: bool,
    need_style: bool,
    need_profile: bool,
    need_glossary: bool,
):
    graph = build_plan_graph(
        need_notes=need_notes,
        need_style=need_style,
        need_profile=need_profile,
        need_glossary=need_glossary,
    )

    sorter = TopologicalSorter(graph)

    return list(sorter.static_order())