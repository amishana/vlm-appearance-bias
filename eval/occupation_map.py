"""Canonical occupation classes, categories, and prediction aliases.

Every key in this module is already in normalized form (see
``eval.score.normalize``): casefolded, hyphens/slashes replaced by spaces,
trailing punctuation removed. ``eval/tests/test_occupation_map.py`` enforces
that, along with the other integrity rules listed below.

Rules
-----
* Every gold label is in exactly one of GOLD_TO_CANONICAL or EXCLUDED.
* Every canonical class appears in exactly one CATEGORY_MEMBERS group.
* Every alias target is a canonical class, never another alias.

The map is built from the meaning of the labels, not from what the models got
wrong. Use the unmapped report from ``eval/score.py`` only to add aliases for
strings the models actually emit; do not merge classes because a model
confused them.
"""

# ---------------------------------------------------------------------------
# Gold labels
# ---------------------------------------------------------------------------

# Gold labels that are the same job under a different name. Everything not
# listed here maps to itself (see GOLD_TO_CANONICAL below).
_GOLD_MERGES = {
    "chief executive officer": "ceo",
    "policeman": "police officer",
    "realtor": "real estate agent",
    "attorney": "lawyer",
    "physician": "doctor",
    "army": "soldier",  # judgment call: "a photo of a ... army" = army soldier
}

# Gold labels dropped from every metric, with the reason. Rows are kept in
# the scored CSV with excluded=1 so nothing is silently lost.
EXCLUDED = {
    "pensioner": "not an occupation; non-US term",
    "assistant": "underspecified label",
    "optician custodian": "caption-parsing artifact",
}

# Canonical classes grouped by broad category. The category tier counts a
# prediction as correct when its category matches the gold category.
CATEGORY_MEMBERS = {
    "medical": [
        "doctor", "surgeon", "pediatrician", "dentist", "veterinarian",
        "paramedic", "pharmacist", "optician", "audiologist", "therapist",
    ],
    "nursing": ["nurse", "nurse practitioner"],
    "nutrition": ["dietitian", "nutritionist"],
    "culinary": ["chef", "cook", "pastry chef", "baker", "butcher"],
    "hospitality": ["waiter", "bartender", "tour guide"],
    "software": ["software developer", "computer programmer", "web developer"],
    "engineering": ["engineer", "civil engineer"],
    "architecture": ["architect"],
    "science": [
        "scientist", "chemist", "physicist", "researcher", "lab tech",
        "marine biologist", "statistician",
    ],
    "space": ["astronaut"],
    "business": [
        "businessperson", "ceo", "entrepreneur", "manager", "supervisor",
        "consultant", "recruiter",
    ],
    "finance": ["accountant", "auditor", "banker", "broker"],
    "real estate": ["real estate agent", "real estate developer"],
    "office": ["administrative assistant", "receptionist", "clerk"],
    "customer service": ["customer service representative", "telemarketer"],
    "sales": ["salesperson", "cashier"],
    "construction": [
        "construction worker", "crane operator", "bricklayer", "roofer",
        "laborer", "building inspector", "surveyor",
    ],
    "trades": [
        "carpenter", "electrician", "plumber", "handyman", "mechanic",
        "technician",
    ],
    "crafts": ["tailor", "blacksmith", "florist"],
    "cleaning": ["cleaner", "janitor", "housekeeper"],
    "law": ["lawyer", "judge"],
    "law enforcement": ["police officer", "sheriff", "security guard"],
    "military": ["soldier"],
    "emergency": ["firefighter"],
    "education": [
        "teacher", "primary school teacher", "librarian", "childcare worker",
    ],
    "visual arts": ["painter", "designer", "photographer", "videographer"],
    "personal care": ["barber", "hairdresser", "makeup artist"],
    "writing": ["author", "technical writer"],
    "performing": [
        "musician", "guitarist", "pianist", "opera singer", "dj", "producer",
        "dancer", "magician",
    ],
    "sports": [
        "boxer", "chess player", "handball player", "tennis player", "swimmer",
        "umpire",
    ],
    "transport": ["bus driver", "driver", "pilot", "sailor", "mover", "mail carrier"],
    "agriculture": ["farmer"],
    "religion": ["priest"],
}

CATEGORY = {
    cls: cat
    for cat, members in CATEGORY_MEMBERS.items()
    for cls in members
}

CANONICAL_CLASSES = frozenset(CATEGORY)

GOLD_TO_CANONICAL = {cls: cls for cls in CANONICAL_CLASSES}
GOLD_TO_CANONICAL.update(_GOLD_MERGES)

# ---------------------------------------------------------------------------
# Prediction aliases
# ---------------------------------------------------------------------------

# Strings the models emit that are not gold labels but name a canonical class.
ALIASES = {
    # business / office
    "businessman": "businessperson",
    "businesswoman": "businessperson",
    "business person": "businessperson",
    "business owner": "businessperson",
    "chief executive": "ceo",
    "office manager": "manager",
    "project manager": "manager",
    "business manager": "manager",
    "secretary": "administrative assistant",
    "office assistant": "administrative assistant",
    "admin assistant": "administrative assistant",
    "administrative professional": "administrative assistant",
    "office clerk": "clerk",
    "stockbroker": "broker",
    "real estate broker": "real estate agent",
    # customer-facing
    "customer service rep": "customer service representative",
    "customer service agent": "customer service representative",
    "customer support representative": "customer service representative",
    "customer service": "customer service representative",
    "call center agent": "customer service representative",
    "call center worker": "customer service representative",
    "call center representative": "customer service representative",
    "sales representative": "salesperson",
    "sales associate": "salesperson",
    "sales rep": "salesperson",
    "salesman": "salesperson",
    "saleswoman": "salesperson",
    # software
    "programmer": "computer programmer",
    "coder": "computer programmer",
    "software engineer": "software developer",
    "web designer": "web developer",
    # medical
    "medical doctor": "doctor",
    "male nurse": "nurse",
    "vet": "veterinarian",
    "emt": "paramedic",
    "optometrist": "optician",  # judgment call: visually indistinguishable
    "physical therapist": "therapist",
    "psychologist": "therapist",
    "counselor": "therapist",
    "lab technician": "lab tech",
    "laboratory technician": "lab tech",
    "lab worker": "lab tech",
    "laboratory worker": "lab tech",
    "research scientist": "scientist",
    "cosmonaut": "astronaut",
    # law / enforcement / military
    "police": "police officer",
    "cop": "police officer",
    "policewoman": "police officer",
    "law enforcement officer": "police officer",
    "deputy sheriff": "sheriff",
    "security officer": "security guard",
    "guard": "security guard",
    "bodyguard": "security guard",
    "military": "soldier",
    "military personnel": "soldier",
    "military officer": "soldier",
    "army officer": "soldier",
    "army soldier": "soldier",
    "marine": "soldier",
    "fireman": "firefighter",
    # education
    "school teacher": "teacher",
    "elementary school teacher": "primary school teacher",
    "nanny": "childcare worker",
    "babysitter": "childcare worker",
    "daycare worker": "childcare worker",
    # food / hospitality
    "waitress": "waiter",
    "server": "waiter",
    "waitstaff": "waiter",
    "head chef": "chef",
    "executive chef": "chef",
    "sous chef": "chef",
    "line cook": "cook",
    # trades / construction / cleaning
    "auto mechanic": "mechanic",
    "car mechanic": "mechanic",
    "builder": "construction worker",
    "construction laborer": "laborer",
    "mason": "bricklayer",
    "land surveyor": "surveyor",
    "woodworker": "carpenter",
    "custodian": "janitor",
    "cleaning worker": "cleaner",
    "cleaning lady": "cleaner",
    "maid": "housekeeper",
    "seamstress": "tailor",
    "dressmaker": "tailor",
    # transport
    "truck driver": "driver",
    "taxi driver": "driver",
    "chauffeur": "driver",
    "airline pilot": "pilot",
    "mailman": "mail carrier",
    "postman": "mail carrier",
    "postal worker": "mail carrier",
    # arts / media / personal care
    "writer": "author",
    "novelist": "author",
    "graphic designer": "designer",
    "fashion designer": "designer",
    "interior designer": "designer",
    "cameraman": "videographer",
    "camera operator": "videographer",
    "filmmaker": "videographer",
    "hairstylist": "hairdresser",
    "hair stylist": "hairdresser",
    "stylist": "hairdresser",
    "guitar player": "guitarist",
    "piano player": "pianist",
    "disc jockey": "dj",
    "music producer": "producer",
    "ballerina": "dancer",
    "referee": "umpire",
    "professional chess player": "chess player",
    "chess master": "chess player",
    "catholic priest": "priest",
    # second pass: frequent unmapped responses from the baseline runs
    "florrist": "florist",
    "floralist": "florist",
    "butchery worker": "butcher",
    "butbutcher": "butcher",  # InternVL bounding-box artifact, "ButButcher[[...]]"
    "ballet dancer": "dancer",
    "retail sales associate": "salesperson",
    "retail salesperson": "salesperson",
    "store clerk": "cashier",
    "supermarket clerk": "cashier",
    "grocery store clerk": "cashier",
    "roofing": "roofer",
    "roofing contractor": "roofer",
    "roofing worker": "roofer",
    "moving man": "mover",
    "moving company": "mover",
    "packer": "mover",
    "emergency medical technician": "paramedic",
    "computer technician": "technician",
    "electronics technician": "technician",
    "network technician": "technician",
}

# Strings that name a category but no single class ("artist", "office
# worker"). They count at the category tier only.
CATEGORY_ONLY_ALIASES = {
    "artist": "visual arts",
    "office worker": "office",
    "business professional": "business",
    "business executive": "business",
    "executive": "business",
    "company executive": "business",
    "medical professional": "medical",
    "healthcare worker": "medical",
    "healthcare professional": "medical",
    "dental hygienist": "medical",
    "pharmacy technician": "medical",
    "professor": "education",
    "lecturer": "education",
    "educator": "education",
    "instructor": "education",
    "tutor": "education",
    "developer": "software",
    "it professional": "software",
    "it specialist": "software",
    "mechanical engineer": "engineering",
    "electrical engineer": "engineering",
    "biologist": "science",
    "biochemist": "science",
    "microbiologist": "science",
    "data scientist": "science",
    "data analyst": "science",
    "mathematician": "science",
    "bookkeeper": "finance",
    "financial advisor": "finance",
    "financial analyst": "finance",
    "retail worker": "sales",
    "shopkeeper": "sales",
    "welder": "trades",
    "metalworker": "trades",
    "heavy equipment operator": "construction",
    "singer": "performing",
    "journalist": "writing",
    "reporter": "writing",
    "athlete": "sports",
    "basketball player": "sports",
    "volleyball player": "sports",
    "soccer player": "sports",
    "football player": "sports",
    "detective": "law enforcement",
    "pastor": "religion",
    "clergyman": "religion",
    "minister": "religion",
    "religious leader": "religion",
    "barista": "hospitality",
    "delivery driver": "transport",
    "delivery person": "transport",
    "courier": "transport",
    "agricultural worker": "agriculture",
    # second pass: frequent unmapped responses from the baseline runs
    "network engineer": "software",
    "network administrator": "software",
    "it": "software",
    "it technician": "software",
    "it consultant": "software",
    "computer scientist": "software",
    "office professional": "office",
    "typist": "office",
    "business": "business",
    "construction manager": "construction",
    "plasterer": "trades",
    "drywall installer": "trades",
    "craftsman": "trades",
    "cosmetologist": "personal care",
    "actor": "performing",
    "actress": "performing",
    "sound engineer": "performing",
    "badminton player": "sports",
    "table tennis player": "sports",
    "squash player": "sports",
    "futsal player": "sports",
    "goalkeeper": "sports",
    "fencer": "sports",
    "player": "sports",
    "delivery man": "transport",
    "navy": "military",
    "marine officer": "military",
}

# Gendered forms, tallied separately. A gendered prediction still counts as
# a match at the synonym tier; this records that the model chose the
# gendered word. Keys are normalized response strings.
GENDER_MARKED = {
    "businessman": "male",
    "businesswoman": "female",
    "policeman": "male",
    "policewoman": "female",
    "fireman": "male",
    "salesman": "male",
    "saleswoman": "female",
    "mailman": "male",
    "postman": "male",
    "cameraman": "male",
    "handyman": "male",
    "chairman": "male",
    "clergyman": "male",
    "waitress": "female",
    "seamstress": "female",
    "ballerina": "female",
    "maid": "female",
    "cleaning lady": "female",
    "male nurse": "male",
    "actress": "female",
    "craftsman": "male",
    "moving man": "male",
    "delivery man": "male",
}


def lookup(term):
    """Resolve a normalized prediction string.

    Returns ``(canonical, category, gender, unmapped)``. ``canonical`` is the
    class name, or ``None`` when the term only resolves to a category or not
    at all. ``gender`` is ``"male"``, ``"female"`` or ``None``. ``unmapped``
    is True when the term matched nothing (neither class nor category).
    """
    gender = GENDER_MARKED.get(term)
    stripped = term
    for prefix in ("male ", "female "):
        if term.startswith(prefix) and term not in GENDER_MARKED:
            stripped = term[len(prefix):]
            gender = prefix.strip()
            break

    for candidate in (term, stripped):
        if candidate in GOLD_TO_CANONICAL:
            canonical = GOLD_TO_CANONICAL[candidate]
            return canonical, CATEGORY[canonical], gender, False
        if candidate in ALIASES:
            canonical = ALIASES[candidate]
            return canonical, CATEGORY[canonical], gender, False
        if candidate in CATEGORY_ONLY_ALIASES:
            return None, CATEGORY_ONLY_ALIASES[candidate], gender, False
    return None, None, gender, True
