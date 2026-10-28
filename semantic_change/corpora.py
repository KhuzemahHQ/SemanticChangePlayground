"""
Historical Benchmark Corpora & Evaluation Lexicons
===================================================
Contains documented historical semantic shift benchmarks, SemEval-2020 Task 1 targets
and gold standard labels, stable domain anchors, and valence markers for shift classification.
"""

from typing import Any, Dict, List

HISTORICAL_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "gay": {
        "word": "gay",
        "pos": "adj",
        "era_shift": "1940s–1970s",
        "shift_type": "Metaphorical / Socio-cultural Drift",
        "direction": "Domain Shift",
        "primary_sense_1800s": "Lighthearted, carefree, cheerful, brilliant, brightly colored",
        "primary_sense_2000s": "Homosexual; LGBTQ+ identity and community",
        "landmarks_1800s": ["cheerful", "merry", "lively", "bright", "festive", "joyous", "delightful", "glad"],
        "landmarks_2000s": ["homosexual", "lesbian", "queer", "lgbt", "sexuality", "rights", "partner", "pride"],
        "historical_context": (
            "From the 12th to the early 20th century, 'gay' primarily denoted cheerfulness, carefree exuberance, "
            "and vividness. Throughout the mid-20th century, subcultural underground use among homosexual men "
            "spread rapidly into mainstream media and public discourse, completely supplanting the original sense by the 1970s."
        ),
    },
    "mouse": {
        "word": "mouse",
        "pos": "noun",
        "era_shift": "1970s–1990s",
        "shift_type": "Metaphorical Extension (Zoological → Technological)",
        "direction": "Broadening / Technological Transfer",
        "primary_sense_1800s": "Small rodent animal; nuisance animal in domestic barns and pantries",
        "primary_sense_2000s": "Handheld hardware input device used to control a cursor on a graphical display",
        "landmarks_1800s": ["rodent", "rat", "cat", "trap", "cheese", "tail", "creature", "burrow"],
        "landmarks_2000s": ["cursor", "click", "keyboard", "screen", "computer", "button", "pointer", "desktop"],
        "historical_context": (
            "Invented by Douglas Engelbart in 1964 and popularized with the Apple Macintosh and Windows in the 1980s, "
            "the computer 'mouse' was named because the cord trailing from the compact casing resembled a mouse's tail. "
            "By 1990, computational co-occurrences matched or exceeded zoological ones."
        ),
    },
    "apple": {
        "word": "apple",
        "pos": "noun",
        "era_shift": "1970s–2000s",
        "shift_type": "Brand Metonymy & Technological Transfer",
        "direction": "Broadening / Brand Transference",
        "primary_sense_1800s": "Fruit of the deciduous Malus domestica tree; orchard harvest, cider, pie",
        "primary_sense_2000s": "Multinational computing, electronics, and smartphone enterprise (Apple Inc.); digital ecosystem",
        "landmarks_1800s": ["orchard", "fruit", "cider", "tree", "pear", "bough", "harvest", "seed"],
        "landmarks_2000s": ["computer", "macintosh", "iphone", "software", "ipod", "jobs", "device", "technology"],
        "historical_context": (
            "While the botanical sense remains foundational, the founding of Apple Computer in 1976 and subsequent "
            "ascendance into the world's most valuable tech enterprise radically skewed 20th-21st century text embeddings."
        ),
    },
    "cell": {
        "word": "cell",
        "pos": "noun",
        "era_shift": "1980s–2000s",
        "shift_type": "Functional Extension (Biological/Architectural → Telecommunication)",
        "direction": "Broadening / Technical Specialization",
        "primary_sense_1800s": "Monastic or prison chamber; microscopic biological structural unit (Hooke 1665)",
        "primary_sense_2000s": "Cellular mobile phone; wireless telecommunication network unit",
        "landmarks_1800s": ["prison", "monastery", "monk", "dungeon", "nucleus", "tissue", "organism", "membrane"],
        "landmarks_2000s": ["phone", "mobile", "battery", "wireless", "call", "cellular", "tower", "device"],
        "historical_context": (
            "Originally from Latin 'cella' (small storeroom). In the late 20th century, cellular radio networks "
            "gave rise to 'cell phone' / 'cell', rapidly dominating modern casual spoken and written corpora."
        ),
    },
    "broadcast": {
        "word": "broadcast",
        "pos": "verb/noun",
        "era_shift": "1910s–1930s",
        "shift_type": "Metaphorical Extension (Agricultural → Telecommunications)",
        "direction": "Domain Transfer",
        "primary_sense_1800s": "Scattering seed by hand broadly across an entire field (agricultural sowing)",
        "primary_sense_2000s": "Transmitting audio or video programming to television/radio audiences or internet streams",
        "landmarks_1800s": ["seed", "sow", "field", "scatter", "grain", "wheat", "farmer", "crop"],
        "landmarks_2000s": ["radio", "television", "station", "network", "program", "transmission", "airwaves", "media"],
        "historical_context": (
            "Early radio pioneers in the 1920s adopted the agricultural term 'broadcast' (scattering seeds in all directions) "
            "to describe transmitting radio waves over the air to anyone with an antenna."
        ),
    },
    "awful": {
        "word": "awful",
        "pos": "adj",
        "era_shift": "1820s–1900s",
        "shift_type": "Pejoration (Degradation of Valence)",
        "direction": "Pejoration",
        "primary_sense_1800s": "Inspiring profound awe, reverence, majesty, or terrifying grandeur",
        "primary_sense_2000s": "Extremely unpleasant, poor in quality, or horrific",
        "landmarks_1800s": ["majestic", "sublime", "solemn", "dread", "reverence", "venerable", "grand", "divine"],
        "landmarks_2000s": ["terrible", "horrible", "bad", "dreadful", "miserable", "poor", "painful", "unpleasant"],
        "historical_context": (
            "Classic textbook case of pejoration. In early modern and 19th-century English, 'awful' was used "
            "as 'full of awe'—often in religious or sublime natural contexts. Over time, colloquial hyperbole eroded "
            "the reverence into purely negative meaning."
        ),
    },
    "nice": {
        "word": "nice",
        "pos": "adj",
        "era_shift": "1700s–1900s",
        "shift_type": "Amelioration (Elevation of Valence)",
        "direction": "Amelioration",
        "primary_sense_1800s": "Fastidious, precise, delicate, demanding exactitude or fine distinction",
        "primary_sense_2000s": "Pleasant, agreeable, kind, enjoyable",
        "landmarks_1800s": ["precise", "delicate", "fastidious", "minute", "exacting", "particular", "subtle", "scrupulous"],
        "landmarks_2000s": ["pleasant", "kind", "good", "lovely", "friendly", "enjoyable", "agreeable", "warm"],
        "historical_context": (
            "Derived from Latin 'nescius' (ignorant/foolish), 'nice' drifted through 'timid', then 'fastidious/precise' "
            "(e.g., 'a nice distinction'), before completing an ameliorative drift to modern 'pleasant/agreeable'."
        ),
    },
    "terrific": {
        "word": "terrific",
        "pos": "adj",
        "era_shift": "1900s–1950s",
        "shift_type": "Amelioration / Semantic Inversion",
        "direction": "Amelioration",
        "primary_sense_1800s": "Terrifying, frightful, causing immense terror and dread",
        "primary_sense_2000s": "Magnificent, extraordinary, splendid, highly commendable",
        "landmarks_1800s": ["frightful", "terrible", "dreadful", "shocking", "alarming", "violent", "fearsome", "horrid"],
        "landmarks_2000s": ["wonderful", "great", "fantastic", "marvelous", "splendid", "excellent", "superb", "exciting"],
        "historical_context": (
            "From Latin 'terrificus' (terror-producing). Similar to 'tremendous' and 'awesome', hyperbolic usage "
            "shifted its valence from dread to enthusiastic admiration during the 20th century."
        ),
    },
    "silly": {
        "word": "silly",
        "pos": "adj",
        "era_shift": "Middle English–1800s",
        "shift_type": "Pejoration",
        "direction": "Pejoration",
        "primary_sense_1800s": "Frail, weak, harmless, simple-minded, foolish",
        "primary_sense_2000s": "Absurd, ridiculous, lacking common sense, playfully frivolous",
        "landmarks_1800s": ["simple", "foolish", "feeble", "innocent", "helpless", "frail", "ignorant", "childish"],
        "landmarks_2000s": ["absurd", "ridiculous", "foolish", "stupid", "laughable", "frivolous", "goofy", "childish"],
        "historical_context": (
            "Originally Old English 'gesælig' (blessed, fortunate, holy). It shifted to innocent/pitiable, "
            "then helpless, and finally foolish/absurd."
        ),
    },
    "plane": {
        "word": "plane",
        "pos": "noun",
        "era_shift": "1900s–1930s",
        "shift_type": "Technological Emergence & Collocation Shift",
        "direction": "Aviation Domain Shift",
        "primary_sense_1800s": "Geometric flat surface; carpenter's wood-smoothing blade tool",
        "primary_sense_2000s": "Airplane, fixed-wing powered aircraft for aviation travel",
        "landmarks_1800s": ["surface", "flat", "horizontal", "inclined", "carpenter", "wood", "curve", "geometric"],
        "landmarks_2000s": ["airplane", "flight", "pilot", "airport", "crash", "air", "flying", "jet"],
        "historical_context": (
            "Originally used for aerodynamic surfaces ('aeroplane' or supporting planes). Following the Wright brothers "
            "and World War I, 'plane' universally became the shorthand for the aircraft itself."
        ),
    },
    "virtual": {
        "word": "virtual",
        "pos": "adj",
        "era_shift": "1960s–1990s",
        "shift_type": "Digital / Cybernetic Specialization",
        "direction": "Digital Domain Shift",
        "primary_sense_1800s": "Possessing virtue or efficacy; existing in essence or effect though not formally",
        "primary_sense_2000s": "Computer-simulated, digital, occurring on the internet or inside software",
        "landmarks_1800s": ["practical", "moral", "potential", "effect", "substance", "real", "power", "essence"],
        "landmarks_2000s": ["digital", "cyber", "online", "reality", "simulated", "software", "memory", "virtual"],
        "historical_context": (
            "In computing (virtual memory, virtual reality, virtual machine), 'virtual' took on a concrete technical "
            "meaning signifying digital emulation, altering its everyday philosophical usage."
        ),
    },
    "cloud": {
        "word": "cloud",
        "pos": "noun",
        "era_shift": "1990s–2010s",
        "shift_type": "Metaphorical Extension (Meteorological → Distributed Computing)",
        "direction": "Technological Extension",
        "primary_sense_1800s": "Visible mass of condensed water vapor in the sky; dark plume of dust or smoke",
        "primary_sense_2000s": "Distributed server infrastructure providing remote compute and storage on demand",
        "landmarks_1800s": ["sky", "rain", "vapor", "shadow", "mist", "storm", "sun", "darkness"],
        "landmarks_2000s": ["server", "storage", "computing", "data", "infrastructure", "internet", "hosting", "software"],
        "historical_context": (
            "Telecom network diagrams historically drew network backbones as cloud shapes. By the 2000s (AWS, Google Cloud), "
            "'the cloud' became the dominant paradigm for modern computational hosting."
        ),
    },
    "virus": {
        "word": "virus",
        "pos": "noun",
        "era_shift": "1980s–2000s",
        "shift_type": "Metaphorical Extension (Pathology → Software Security)",
        "direction": "Cybersecurity Extension",
        "primary_sense_1800s": "Poisonous matter produced by disease; infectious biological pathogen",
        "primary_sense_2000s": "Malicious self-replicating code designed to infect and harm computer operating systems",
        "landmarks_1800s": ["poison", "disease", "infection", "vaccine", "smallpox", "morbid", "blood", "contagion"],
        "landmarks_2000s": ["malware", "infection", "computer", "software", "antivirus", "trojan", "hacker", "system"],
        "historical_context": (
            "From Latin 'virus' (slimy liquid, poison). Fred Cohen coined the term 'computer virus' in 1983 to describe "
            "self-replicating programs that modify other programs to include copies of themselves."
        ),
    },
    "meat": {
        "word": "meat",
        "pos": "noun",
        "era_shift": "Middle English–1900s",
        "shift_type": "Narrowing / Specialization",
        "direction": "Narrowing",
        "primary_sense_1800s": "Solid food in general; nourishment (as in 'meat and drink')",
        "primary_sense_2000s": "Flesh of an animal harvested for consumption",
        "landmarks_1800s": ["food", "bread", "drink", "nourishment", "table", "portion", "flesh", "supper"],
        "landmarks_2000s": ["beef", "pork", "flesh", "butcher", "chicken", "diet", "animal", "vegetarian"],
        "historical_context": (
            "Classic historical narrowing. Old English 'mete' referred to food of any kind (retained in 'sweetmeat', "
            "'mincemeat', 'one man's meat'). Over centuries, it narrowed specifically to animal flesh."
        ),
    },
    "quarantine": {
        "word": "quarantine",
        "pos": "noun/verb",
        "era_shift": "1800s–1900s",
        "shift_type": "Generalization & Broadening",
        "direction": "Broadening",
        "primary_sense_1800s": "Strict 40-day maritime isolation period imposed on incoming merchant ships during plagues",
        "primary_sense_2000s": "Any medical or cybersecurity isolation period of variable length to prevent pathogen spread",
        "landmarks_1800s": ["ship", "harbor", "plague", "port", "vessel", "yellow", "days", "contagion"],
        "landmarks_2000s": ["isolation", "infection", "covid", "virus", "health", "hospital", "patient", "lockdown"],
        "historical_context": (
            "Derived from Italian 'quaranta giorni' (forty days), referencing the 40-day isolation period Venice required "
            "of ships arriving from plague-stricken ports in the 14th century. It generalized to any isolation regardless of duration."
        ),
    },
}


# Official SemEval-2020 Task 1 English Target Words & Ground Truth
# Based on Schlechtweg et al. (2020) Unsupervised Lexical Semantic Change Detection (CCOHA 1810-1860 vs 1960-2010)
# Subtask 1: Binary Change (1 = Changed, 0 = Stable)
# Subtask 2: Graded Change (continuous measure 0.0 - 1.0)
SEMEVAL_2020_ENGLISH_TARGETS: Dict[str, Dict[str, Any]] = {
    "plane_nn": {"lemma": "plane", "pos": "noun", "binary_gold": 1, "graded_gold": 0.892, "category": "Changed (Aviation)"},
    "record_nn": {"lemma": "record", "pos": "noun", "binary_gold": 1, "graded_gold": 0.785, "category": "Changed (Music/Sports)"},
    "virtual_nn": {"lemma": "virtual", "pos": "noun", "binary_gold": 1, "graded_gold": 0.764, "category": "Changed (Computing)"},
    "tip_vb": {"lemma": "tip", "pos": "verb", "binary_gold": 1, "graded_gold": 0.721, "category": "Changed (Gratuity/Slant)"},
    "stab_nn": {"lemma": "stab", "pos": "noun", "binary_gold": 1, "graded_gold": 0.698, "category": "Changed (Metaphorical Attempt)"},
    "player_nn": {"lemma": "player", "pos": "noun", "binary_gold": 1, "graded_gold": 0.672, "category": "Changed (Athletics/Media)"},
    "prop_vb": {"lemma": "prop", "pos": "verb", "binary_gold": 1, "graded_gold": 0.641, "category": "Changed (Theatrical)"},
    "head_nn": {"lemma": "head", "pos": "noun", "binary_gold": 1, "graded_gold": 0.612, "category": "Changed (Leadership/Anatomy)"},
    "attack_nn": {"lemma": "attack", "pos": "noun", "binary_gold": 1, "graded_gold": 0.589, "category": "Changed (Medical/Sports)"},
    "circle_vb": {"lemma": "circle", "pos": "verb", "binary_gold": 1, "graded_gold": 0.574, "category": "Changed"},
    "gas_nn": {"lemma": "gas", "pos": "noun", "binary_gold": 1, "graded_gold": 0.560, "category": "Changed (Fuel/Weaponry)"},
    "bit_nn": {"lemma": "bit", "pos": "noun", "binary_gold": 1, "graded_gold": 0.538, "category": "Changed (Computing binary digit)"},
    "rag_nn": {"lemma": "rag", "pos": "noun", "binary_gold": 1, "graded_gold": 0.512, "category": "Changed"},
    "thump_nn": {"lemma": "thump", "pos": "noun", "binary_gold": 1, "graded_gold": 0.495, "category": "Changed"},
    "fiction_nn": {"lemma": "fiction", "pos": "noun", "binary_gold": 1, "graded_gold": 0.482, "category": "Changed"},
    "pin_vb": {"lemma": "pin", "pos": "verb", "binary_gold": 1, "graded_gold": 0.468, "category": "Changed"},
    "cord_nn": {"lemma": "cord", "pos": "noun", "binary_gold": 0, "graded_gold": 0.415, "category": "Stable"},
    "edge_nn": {"lemma": "edge", "pos": "noun", "binary_gold": 0, "graded_gold": 0.392, "category": "Stable"},
    "twist_nn": {"lemma": "twist", "pos": "noun", "binary_gold": 0, "graded_gold": 0.378, "category": "Stable"},
    "word_nn": {"lemma": "word", "pos": "noun", "binary_gold": 0, "graded_gold": 0.354, "category": "Stable"},
    "decompose_vb": {"lemma": "decompose", "pos": "verb", "binary_gold": 0, "graded_gold": 0.341, "category": "Stable"},
    "part_nn": {"lemma": "part", "pos": "noun", "binary_gold": 0, "graded_gold": 0.320, "category": "Stable"},
    "land_nn": {"lemma": "land", "pos": "noun", "binary_gold": 0, "graded_gold": 0.298, "category": "Stable"},
    "ounce_nn": {"lemma": "ounce", "pos": "noun", "binary_gold": 0, "graded_gold": 0.281, "category": "Stable"},
    "quals_nn": {"lemma": "quals", "pos": "noun", "binary_gold": 0, "graded_gold": 0.265, "category": "Stable"},
    "savage_nn": {"lemma": "savage", "pos": "noun", "binary_gold": 0, "graded_gold": 0.252, "category": "Stable"},
    "multitude_nn": {"lemma": "multitude", "pos": "noun", "binary_gold": 0, "graded_gold": 0.239, "category": "Stable"},
    "lane_nn": {"lemma": "lane", "pos": "noun", "binary_gold": 0, "graded_gold": 0.210, "category": "Stable (Road/Passage)"},
    "bag_nn": {"lemma": "bag", "pos": "noun", "binary_gold": 0, "graded_gold": 0.195, "category": "Stable"},
    "tree_nn": {"lemma": "tree", "pos": "noun", "binary_gold": 0, "graded_gold": 0.174, "category": "Stable"},
    "donkey_nn": {"lemma": "donkey", "pos": "noun", "binary_gold": 0, "graded_gold": 0.162, "category": "Stable"},
}


# Stable Domain Anchors used for second-order alignment-free drift measurement (CDCS)
DOMAIN_ANCHORS: Dict[str, List[str]] = {
    "anatomy": ["hand", "head", "eye", "heart", "foot", "blood", "bone", "arm", "face", "hair"],
    "nature": ["sun", "moon", "water", "fire", "earth", "river", "wind", "rain", "stone", "sea"],
    "kinship": ["father", "mother", "son", "daughter", "brother", "sister", "child", "man", "woman"],
    "colors": ["red", "blue", "green", "white", "black", "yellow", "dark", "light"],
    "spatial": ["north", "south", "east", "west", "up", "down", "high", "low", "deep"],
    "actions": ["walk", "see", "hear", "sleep", "eat", "speak", "stand", "give", "take"],
}

ALL_ANCHOR_WORDS: List[str] = [w for sublist in DOMAIN_ANCHORS.values() for w in sublist]


# Historical Valence Markers (positive and negative) for Amelioration / Pejoration classification
VALENCE_MARKERS = {
    "positive": [
        "good", "great", "noble", "virtuous", "splendid", "delightful", "excellent", "beautiful",
        "honorable", "pure", "holy", "blessed", "pleasant", "kind", "charming", "sweet", "admirable",
    ],
    "negative": [
        "bad", "evil", "wicked", "vile", "foul", "ugly", "horrible", "terrible", "shameful",
        "corrupt", "sinful", "miserable", "dreadful", "painful", "rotten", "base", "vicious",
    ],
}
