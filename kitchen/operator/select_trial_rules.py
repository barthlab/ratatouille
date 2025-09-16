
from typing import Dict
from kitchen.settings.timeline import REWARD_EVENTS_DEFAULT, STIMULUS_EVENTS_DEFAULT
from kitchen.structure.neural_data_structure import Timeline

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
    "ClassicPuff": {
        "timeline": lambda x: (len(x.filter("Puff")) == 1) & (len(x.filter(REWARD_EVENTS_DEFAULT)) == 0),
    },
    "ClassicBlank": {
        "timeline": lambda x: (len(x.filter("Blank")) == 1) & (len(x.filter(REWARD_EVENTS_DEFAULT)) == 0),
    },
    "FakeRelayNoWater": {
        "timeline": lambda x: (len(x.filter("NoWaterOn")) == 1) & (len(x.filter("FakeRelayOn")) == 1),
    },
    "FakeRelayWater": {
        "timeline": lambda x: (len(x.filter("WaterOn")) == 1) & (len(x.filter("FakeRelayOn")) == 1),
    },
    "PeltierLeftWater": {
        "timeline": lambda x: (len(x.filter("PeltierLeftOn")) == 1) & (len(x.filter("WaterOn")) == 1),
    },
    "PeltierRightNoWater": {
        "timeline": lambda x: (len(x.filter("PeltierRightOn")) == 1) & (len(x.filter("NoWaterOn")) == 1),
    },
    "PeltierBothWater": {
        "timeline": lambda x: (len(x.filter("PeltierBothOn")) == 1) & (len(x.filter("WaterOn")) == 1),
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


def _puff_duration(timeline: Timeline) -> float:
    puff_on = timeline.filter("VerticalPuffOn")
    puff_off = timeline.filter("VerticalPuffOff")
    assert len(puff_on) == 1, f"Expected 1 puff on, got {len(puff_on)}"
    assert len(puff_off) == 1, f"Expected 1 puff off, got {len(puff_off)}"
    return float(puff_off.t[0] - puff_on.t[0])


PREDEFINED_PASSIVEPUFF_RULES = {
    "500msPuff": {
        "timeline": lambda x: 0.48 <_puff_duration(x) < 0.52,
        "hash_key": "trial"
    },
    "100msPuff": {
        "timeline": lambda x: 0.08 <_puff_duration(x) < 0.12,
        "hash_key": "trial"
    },
}





