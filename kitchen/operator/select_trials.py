
from typing import Dict
from kitchen.settings.timeline import STIMULUS_EVENTS_DEFAULT
from kitchen.structure.hierarchical_data_structure import DataSet

# pre-defined rules
PREDEFINED_RULES = {
    "OnlyWater": {
        "hash_key": "trial",
        "timeline": lambda x: (len(x.filter("WaterOn")) > 0) & (len(x.filter(STIMULUS_EVENTS_DEFAULT)) == 0)
    },
    "OnlyNoWater": {
        "hash_key": "trial",
        "timeline": lambda x: (len(x.filter("NoWaterOn")) > 0) & (len(x.filter(STIMULUS_EVENTS_DEFAULT)) == 0)
    },
    "PuffWater": {
        "hash_key": "trial",
        "timeline": lambda x: (len(x.filter("WaterOn")) > 0) & (len(x.filter("VerticalPuffOn")) > 0)
    },
    "PuffNoWater": {
        "hash_key": "trial",
        "timeline": lambda x: (len(x.filter("NoWaterOn")) > 0) & (len(x.filter("VerticalPuffOn")) > 0)
    },
    "BlankNoWater": {
        "hash_key": "trial",
        "timeline": lambda x: (len(x.filter("NoWaterOn")) > 0) & (len(x.filter("BlankOn")) > 0)
    }
}


def select_predefined_trial_types(dataset: DataSet) -> Dict[str, DataSet]:
    """Select trials based on pre-defined rules."""
    
    selected_trials = {}    
    for name, rule_definition in PREDEFINED_RULES.items():
        this_type_trials = dataset.select(**rule_definition)
        if len(this_type_trials) > 0:
            selected_trials[name] = this_type_trials            
    return selected_trials



