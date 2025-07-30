
from typing import Dict
from kitchen.settings.timeline import REWARD_EVENTS_DEFAULT, STIMULUS_EVENTS_DEFAULT

# pre-defined rules
PREDEFINED_RULES: Dict[str, dict] = {
    "OnlyWater": {
        "timeline": lambda x: (len(x.filter("WaterOn")) == 1) & (len(x.filter(STIMULUS_EVENTS_DEFAULT)) == 0),
    },
    "OnlyNoWater": {
        "timeline": lambda x: (len(x.filter("NoWaterOn")) == 1) & (len(x.filter(STIMULUS_EVENTS_DEFAULT)) == 0),
    },
    "PuffWater": {
        "timeline": lambda x: (len(x.filter("WaterOn")) == 1) & (len(x.filter("VerticalPuffOn")) == 1),
    },
    "PuffNoWater": {
        "timeline": lambda x: (len(x.filter("NoWaterOn")) == 1) & (len(x.filter("VerticalPuffOn")) == 1),
    },
    "BlankNoWater": {
        "timeline": lambda x: (len(x.filter("NoWaterOn")) == 1) & (len(x.filter("BlankOn")) == 1),
    },
    "PurePuff": {
        "timeline": lambda x: (len(x.filter("VerticalPuffOn")) == 1) & (len(x.filter(REWARD_EVENTS_DEFAULT)) == 0),
    },
    "PureBlank": {
        "timeline": lambda x: (len(x.filter("BlankOn")) == 1) & (len(x.filter(REWARD_EVENTS_DEFAULT)) == 0),
    },
}

PREDEFINED_TRIAL_RULES = {
    name: rule_definition | {"hash_key": "trial"}
    for name, rule_definition in PREDEFINED_RULES.items()
}

PREDEFINED_FOVTRIAL_RULES = {
    name: rule_definition | {"hash_key": "fovtrial"}
    for name, rule_definition in PREDEFINED_RULES.items()
}





