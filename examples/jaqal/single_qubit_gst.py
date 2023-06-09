(
    "circuit",
    ("usepulses", "qscout.v1.std", "*"),
    ("register", "q", 1),
    ("macro", "F0", "qubit", ("sequential_block",)),
    ("macro", "F1", "qubit", ("sequential_block", ("gate", "Sx", "qubit"))),
    ("macro", "F2", "qubit", ("sequential_block", ("gate", "Sy", "qubit"))),
    (
        "macro",
        "F3",
        "qubit",
        ("sequential_block", ("gate", "Sx", "qubit"), ("gate", "Sx", "qubit")),
    ),
    (
        "macro",
        "F4",
        "qubit",
        (
            "sequential_block",
            ("gate", "Sx", "qubit"),
            ("gate", "Sx", "qubit"),
            ("gate", "Sx", "qubit"),
        ),
    ),
    (
        "macro",
        "F5",
        "qubit",
        (
            "sequential_block",
            ("gate", "Sy", "qubit"),
            ("gate", "Sy", "qubit"),
            ("gate", "Sy", "qubit"),
        ),
    ),
    ("macro", "G0", "qubit", ("sequential_block", ("gate", "Sx", "qubit"))),
    ("macro", "G1", "qubit", ("sequential_block", ("gate", "Sy", "qubit"))),
    ("macro", "G2", "qubit", ("sequential_block", ("gate", "I_Sx", "qubit"))),
    (
        "macro",
        "G3",
        "qubit",
        ("sequential_block", ("gate", "Sx", "qubit"), ("gate", "Sy", "qubit")),
    ),
    (
        "macro",
        "G4",
        "qubit",
        (
            "sequential_block",
            ("gate", "Sx", "qubit"),
            ("gate", "Sy", "qubit"),
            ("gate", "I_Sx", "qubit"),
        ),
    ),
    (
        "macro",
        "G5",
        "qubit",
        (
            "sequential_block",
            ("gate", "Sx", "qubit"),
            ("gate", "I_Sx", "qubit"),
            ("gate", "Sy", "qubit"),
        ),
    ),
    (
        "macro",
        "G6",
        "qubit",
        (
            "sequential_block",
            ("gate", "Sx", "qubit"),
            ("gate", "I_Sx", "qubit"),
            ("gate", "I_Sx", "qubit"),
        ),
    ),
    (
        "macro",
        "G7",
        "qubit",
        (
            "sequential_block",
            ("gate", "Sy", "qubit"),
            ("gate", "I_Sx", "qubit"),
            ("gate", "I_Sx", "qubit"),
        ),
    ),
    (
        "macro",
        "G8",
        "qubit",
        (
            "sequential_block",
            ("gate", "Sx", "qubit"),
            ("gate", "Sx", "qubit"),
            ("gate", "I_Sx", "qubit"),
            ("gate", "Sy", "qubit"),
        ),
    ),
    (
        "macro",
        "G9",
        "qubit",
        (
            "sequential_block",
            ("gate", "Sx", "qubit"),
            ("gate", "Sy", "qubit"),
            ("gate", "Sy", "qubit"),
            ("gate", "I_Sx", "qubit"),
        ),
    ),
    (
        "macro",
        "G10",
        "qubit",
        (
            "sequential_block",
            ("gate", "Sx", "qubit"),
            ("gate", "Sx", "qubit"),
            ("gate", "Sy", "qubit"),
            ("gate", "Sx", "qubit"),
            ("gate", "Sy", "qubit"),
            ("gate", "Sy", "qubit"),
        ),
    ),
    ("gate", "prepare_all"),
    ("gate", "F0", ("array_item", "q", 0)),
    ("gate", "measure_all"),
    ("gate", "prepare_all"),
    ("gate", "F1", ("array_item", "q", 0)),
    ("gate", "measure_all"),
    ("gate", "prepare_all"),
    ("gate", "F2", ("array_item", "q", 0)),
    ("gate", "measure_all"),
    ("gate", "prepare_all"),
    ("gate", "F3", ("array_item", "q", 0)),
    ("gate", "measure_all"),
    ("gate", "prepare_all"),
    ("gate", "F4", ("array_item", "q", 0)),
    ("gate", "measure_all"),
    ("gate", "prepare_all"),
    ("gate", "F5", ("array_item", "q", 0)),
    ("gate", "measure_all"),
    ("gate", "prepare_all"),
    ("gate", "F1", ("array_item", "q", 0)),
    ("gate", "F1", ("array_item", "q", 0)),
    ("gate", "measure_all"),
    ("gate", "prepare_all"),
    ("gate", "F1", ("array_item", "q", 0)),
    ("gate", "F2", ("array_item", "q", 0)),
    ("gate", "measure_all"),
    ("gate", "prepare_all"),
    ("gate", "F1", ("array_item", "q", 0)),
    ("loop", 8, ("sequential_block", ("gate", "G1", ("array_item", "q", 0)))),
    ("gate", "F1", ("array_item", "q", 0)),
    ("gate", "measure_all"),
)
