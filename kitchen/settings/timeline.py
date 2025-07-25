# Timeline data loading settings and constants


SUPPORTED_TIMELINE_EVENT = {
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
    "task start",
    "task end",
    "Puff",
    "Blank",
}

TTL_EVENT_DEFAULT = (
    "VerticalPuffOn",
    "HorizontalPuffOn",
    "BlankOn",
    "Puff",
    "Blank",
    )

TRIAL_ALIGN_EVENT_DEFAULT = (
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
    )
)

STIMULUS_EVENTS_DEFAULT = (
    "VerticalPuffOn",
    "HorizontalPuffOn",
    "BlankOn",
    "Puff",
    "Blank",
)

REWARD_EVENTS_DEFAULT = (
    "WaterOn",
    "NoWaterOn",
)


ALIGNMENT_STYLE = {
    "Aligned2Stim": STIMULUS_EVENTS_DEFAULT,
    "Aligned2Reward": REWARD_EVENTS_DEFAULT,
    "Aligned2Buzzer": ("BuzzerOn",),
}

