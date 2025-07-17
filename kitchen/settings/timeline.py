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