from reply_graph import get_execution_order


tests = [
    {
        "name": "Notes only",
        "need_notes": True,
        "need_style": False,
        "need_profile": False,
    },
    {
        "name": "Notes + Style",
        "need_notes": True,
        "need_style": True,
        "need_profile": False,
    },
    {
        "name": "Profile only",
        "need_notes": False,
        "need_style": False,
        "need_profile": True,
    },
    {
        "name": "Notes + Style + Profile",
        "need_notes": True,
        "need_style": True,
        "need_profile": True,
    },
]


for test in tests:

    order = get_execution_order(
        need_notes=test["need_notes"],
        need_style=test["need_style"],
        need_profile=test["need_profile"],
    )

    print("\nTEST:")
    print(test["name"])

    print("EXECUTION ORDER:")
    print(order)

    print("-" * 60)