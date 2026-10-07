from graphlib import TopologicalSorter


def build_plan_graph(need_notes: bool, need_style: bool, need_profile: bool):
    """
    Build the dependency graph for the reply process.
    """

    graph = {}

    # Notes retrieval is independent.
    if need_notes:
        graph["notes"] = set()

    # Style retrieval is independent.
    if need_style:
        graph["style"] = set()

    # Profile retrieval is independent.
    if need_profile:
        graph["profile"] = set()

    # Final reply depends on all requested resources.
    graph["reply"] = set(graph.keys())

    return graph


def get_execution_order(
    need_notes: bool,
    need_style: bool,
    need_profile: bool
):
    """
    Return the dependency-respecting execution order.
    """

    graph = build_plan_graph(
        need_notes=need_notes,
        need_style=need_style,
        need_profile=need_profile,
    )

    sorter = TopologicalSorter(graph)

    return list(sorter.static_order())