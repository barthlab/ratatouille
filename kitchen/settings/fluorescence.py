CROP_BEGGINNG = 0.3  # s - crop the first 300ms of the recording (See MES software manual)
DEFAULT_RECORDING_DURATION = 600  # s - the default recording duration for each session
FAST_MATCHING_MODE = True  # only match the first event

NULL_TTL_OFFSET = 0.5  # s - if no ttl is found, align the fluorescence to the timeline by this offset
DF_F0_SIGN = r"$\Delta F/F_0$"


# process fluorescence
DETREND_BASELINE_PERCENTILE = 20  # %
DETREND_BASELINE_WINDOW = 60  # s


# baseline for df/f0
TRIAL_DF_F0_WINDOW = (-1, 0)  # s