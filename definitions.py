ALL_DOMAINS = ["attraction", "hotel", "restaurant", "taxi", "train", "hospital", "police"]

NORMALIZE_SLOT_NAMES = {
    "car type": "car",
    "entrance fee": "price",
    "duration": "time",
    "leaveat": 'leave',
    'arriveby': 'arrive',
    'trainid': 'id'
}

REQUESTABLE_SLOTS = {
    "taxi": ["car", "phone"],
    "police": ["postcode", "address", "phone"],
    "hospital": ["address", "phone", "postcode"],
    "hotel": ["address", "postcode", "internet", "phone", "parking",
              "type", "pricerange", "stars", "area", "reference"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference"],
    "train": ["time", "leave", "price", "arrive", "id", "reference"],
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
}

ALL_REQSLOT = ["car", "address", "postcode", "phone", "internet", "parking", "type", "pricerange", "food",
                      "stars", "area", "reference", "time", "leave", "price", "arrive", "id"]

INFORMABLE_SLOTS = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}

ALL_INFSLOT = ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name",
               "leave", "destination", "departure", "arrive", "department", "food", "time"]

EXTRACTIVE_SLOT = ["leave", "arrive", "destination", "departure", "type", "name", "food"]

DA_ABBR_TO_SLOT_NAME = {
    'addr': "address",
    'fee': "price",
    'post': "postcode",
    'ref': 'reference',
    'ticket': 'price',
    'depart': "departure",
    'dest': "destination",
}

DIALOG_ACTS = {
    'restaurant': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'hotel': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'attraction': ['inform', 'request', 'nooffer', 'recommend', 'select'],
    'train': ['inform', 'request', 'nooffer', 'offerbook', 'offerbooked', 'select'],
    'taxi': ['inform', 'request'],
    'police': ['inform', 'request'],
    'hospital': ['inform', 'request'],
    'general': ['bye', 'greet', 'reqmore', 'welcome'],
}

BOS_USER_TOKEN = "<bos_user>"
EOS_USER_TOKEN = "<eos_user>"

USER_TOKENS = [BOS_USER_TOKEN, EOS_USER_TOKEN]

BOS_BELIEF_TOKEN = "<bos_belief>"
EOS_BELIEF_TOKEN = "<eos_belief>"

BELIEF_TOKENS = [BOS_BELIEF_TOKEN, EOS_BELIEF_TOKEN]

BOS_DB_TOKEN = "<bos_db>"
EOS_DB_TOKEN = "<eos_db>"

DB_TOKENS = [BOS_DB_TOKEN, EOS_DB_TOKEN]

BOS_ACTION_TOKEN = "<bos_act>"
EOS_ACTION_TOKEN = "<eos_act>"

ACTION_TOKENS = [BOS_ACTION_TOKEN, EOS_ACTION_TOKEN]

BOS_RESP_TOKEN = "<bos_resp>"
EOS_RESP_TOKEN = "<eos_resp>"

RESP_TOKENS = [BOS_RESP_TOKEN, EOS_RESP_TOKEN]

DB_NULL_TOKEN = "[db_null]"
DB_0_TOKEN = "[db_0]"
DB_1_TOKEN = "[db_1]"
DB_2_TOKEN = "[db_2]"
DB_3_TOKEN = "[db_3]"

DB_STATE_TOKENS = [DB_NULL_TOKEN, DB_0_TOKEN, DB_1_TOKEN, DB_2_TOKEN, DB_3_TOKEN]

BOS_OPT_TOKEN = "<bos_opt>"
EOS_OPT_TOKEN = "<eos_opt>"
BOS_OPT1_TOKEN = "<bos_opt1>"
EOS_OPT1_TOKEN = "<eos_opt1>"
BOS_OPT2_TOKEN = "<bos_opt2>"
EOS_OPT2_TOKEN = "<eos_opt2>"
OPTION_TOKENS = [BOS_OPT_TOKEN, EOS_OPT_TOKEN, BOS_OPT1_TOKEN, EOS_OPT1_TOKEN, BOS_OPT2_TOKEN, EOS_OPT2_TOKEN]

BOS_OPT_RES_TOKEN = "<bos_opt_res>"
EOS_OPT_RES_TOKEN = "<eos_opt_res>"
OPT1_TOKEN = "[opt1]"
OPT2_TOKEN = "[opt2]"
OPTION_RESULT_TOKENS = [BOS_OPT_RES_TOKEN, EOS_OPT_RES_TOKEN, OPT1_TOKEN, OPT2_TOKEN]

SPECIAL_TOKENS = USER_TOKENS + BELIEF_TOKENS + DB_TOKENS + ACTION_TOKENS + RESP_TOKENS + DB_STATE_TOKENS + OPTION_TOKENS + OPTION_RESULT_TOKENS

MAX_SEQ_LEN = 104
