import numpy as np

from trafficpred.data import files

DAY_OF_WEEK = slice(0, 7)
WEEKEND = 7
TEMPERATURE = 8
WIND = 9
WEATHER = slice(10, 27)

CALENDAR_NAMES = [f"dow_{d}" for d in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")] + \
                 ["weekend", "time_sin", "time_cos"]
WEATHER_NAMES = ["temperature", "wind"] + [f"weather_{k}" for k in range(17)]
NAMES = CALENDAR_NAMES + WEATHER_NAMES
DIM = len(NAMES)
STEPS_PER_DAY = 48

TAXIBJ_FILES = {
    "TaxiBJ/TaxiBJ2013.npy": ("TaxiBJ/TaxiBJext2013.npy", slice(None)),
    "TaxiBJ/TaxiBJ2014.npy": ("TaxiBJ/TaxiBJext2014.npy", slice(None)),
    "TaxiBJ/TaxiBJ2015.npy": ("TaxiBJ/TaxiBJext2015.npy", slice(None)),
    "TaxiBJ/TaxiBJ2016_1.npy": ("TaxiBJ/TaxiBJext2016.npy", slice(0, 3312)),
    "TaxiBJ/TaxiBJ2016_2.npy": ("TaxiBJ/TaxiBJext2016.npy", slice(3312, None)),
}


def raw_for(flow_file, n_frames):
    if flow_file not in TAXIBJ_FILES:
        raise ValueError(f"no external data for {flow_file}")
    ext_file, rows = TAXIBJ_FILES[flow_file]
    raw = np.asarray(files.load(ext_file), dtype=np.float32)[rows]
    if len(raw) != n_frames:
        raise ValueError(f"{ext_file} has {len(raw)} rows for the {n_frames} frames of {flow_file}")
    return raw


def weather_stats(raw_parts):
    stacked = np.concatenate([r[:, [TEMPERATURE, WIND]] for r in raw_parts])
    return stacked.mean(0), stacked.std(0) + 1e-6


def features(raw, mean, std):
    slot = np.arange(len(raw)) % STEPS_PER_DAY
    angle = 2 * np.pi * slot / STEPS_PER_DAY
    calendar = np.concatenate([raw[:, DAY_OF_WEEK], raw[:, [WEEKEND]],
                               np.sin(angle)[:, None], np.cos(angle)[:, None]], axis=1)
    weather = np.concatenate([(raw[:, [TEMPERATURE, WIND]] - mean) / std, raw[:, WEATHER]], axis=1)
    return np.concatenate([calendar, weather], axis=1).astype(np.float32)


def window(feats, i, t_in, t_out):
    w = feats[i:i + t_in + t_out].copy()
    w[t_in:, len(CALENDAR_NAMES):] = w[t_in - 1, len(CALENDAR_NAMES):]
    return w
