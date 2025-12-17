# Timeline data loading settings and constants


SUPPORTED_TIMELINE_EVENT = {
    "TrialOn",
    "VerticalPuffOn",
    "VerticalPuffOff",
    "HorizontalPuffOn",
    "HorizontalPuffOff",
    "BlankOn",
    "BlankOff",
    "WaterOn",
    "WaterOff",
    "NoWaterOn",
    "NoWaterOff",
    "BuzzerOn",
    "BuzzerOff",
    "FakeRelayOn",
    "FakeRelayOff",
    "PeltierLeftOn",
    "PeltierLeftOff",
    "PeltierRightOn",
    "PeltierRightOff",
    "PeltierBothOn",
    "PeltierBothOff",
    "task start",
    "task end",
    "Puff",
    "Blank",
    "Frame",
    "StimOn",
    "StimOff",
    "blueLEDOn",
    "blueLEDOff",
}

TTL_EVENT_DEFAULT = (
    (
        "TrialOn",
    ),
    (
        "BuzzerOn",
    ),
    (        
        "VerticalPuffOn",
        "HorizontalPuffOn",
        "BlankOn",
        "Puff",
        "Blank",
    ),
    (
        "blueLEDOn",
    ),
    (
        "WaterOn",
        "NoWaterOn",
    ),
)

TRIAL_ALIGN_EVENT_DEFAULT = (
    (
        "TrialOn",
    ),
    (
        "VerticalPuffOn",
        "HorizontalPuffOn",
        "BlankOn",
    ),
    (
        "Puff",
        "Blank",
    ),
    (
        "BuzzerOn",
    ),
    (
        "blueLEDOn",
    ),
    (
        "WaterOn",
        "NoWaterOn",
    ),
)

STIMULUS_EVENTS_DEFAULT = (
    "VerticalPuffOn",
    "HorizontalPuffOn",
    "BlankOn",
    "Puff",
    "Blank",
    "PeltierLeftOn",
    "PeltierRightOn",
    "PeltierBothOn",
    "FakeRelayOn",
    "blueLEDOn",
)

REWARD_EVENTS_DEFAULT = (
    "WaterOn",
    "NoWaterOn",
)

AUDITORY_EVENTS_DEFAULT = (
    "BuzzerOn",
)


ALL_ALIGNMENT_STYLE = {
    "Aligned2Stim": STIMULUS_EVENTS_DEFAULT,
    "Aligned2Reward": REWARD_EVENTS_DEFAULT,
    "Aligned2Buzzer": AUDITORY_EVENTS_DEFAULT,
    "Aligned2Trial": ("TrialOn",),
}

