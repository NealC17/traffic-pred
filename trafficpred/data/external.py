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


def calendar(start, n, steps_per_day, holidays=()):
    """Calendar-only features (CALENDAR_NAMES) for n steps from date `start` (YYYY-MM-DD): day of week,
    a day-off flag (weekend or public holiday) and the time of day. Known ahead, so no weather."""
    import datetime as dt
    first = dt.date.fromisoformat(start)
    off = {dt.date.fromisoformat(d) for d in holidays}
    days = [first + dt.timedelta(days=int(k)) for k in np.arange(n) // steps_per_day]
    dow = np.eye(7, dtype=np.float32)[[d.weekday() for d in days]]
    day_off = np.array([d.weekday() >= 5 or d in off for d in days], np.float32)[:, None]
    angle = 2 * np.pi * (np.arange(n) % steps_per_day) / steps_per_day
    return np.concatenate([dow, day_off, np.sin(angle)[:, None], np.cos(angle)[:, None]], 1).astype(np.float32)


NYC_WEATHER_NAMES = ["temperature", "wind", "log_precip", "log_snowfall", "log_snow_depth", "cloud_cover"]


def nyc_weather(json_file, start, n, steps_per_day):
    """(n, 6) raw weather per frame from an Open-Meteo hourly archive file (datasets/weather/*.json, local
    time). Frame f covers [start + f / steps_per_day days, + one step); it gets the hourly record stamped at
    the last full hour at or before the frame's END, so everything in it is known once the frame is over:
    temperature, wind and cloud cover are instantaneous at that hour, precipitation / snowfall are the sums
    over the hour before it (Open-Meteo's convention). Columns: NYC_WEATHER_NAMES, untransformed
    (temperature C, wind km/h, precipitation mm, snowfall cm, snow depth m, cloud cover %)."""
    import datetime as dt
    import json
    h = json.load(open(files.path(json_file)))["hourly"]
    stamps = [dt.datetime.fromisoformat(t) for t in h["time"]]
    index = {t: k for k, t in enumerate(stamps)}
    first = dt.datetime.fromisoformat(start)
    step = dt.timedelta(days=1) / steps_per_day
    cols = ("temperature_2m", "wind_speed_10m", "precipitation", "snowfall", "snow_depth", "cloud_cover")
    out = np.zeros((n, len(cols)), np.float32)
    for f in range(n):
        end = first + (f + 1) * step
        k = index.get(end.replace(minute=0, second=0, microsecond=0))
        if k is None:
            raise ValueError(f"{json_file} has no record for {end} (frame {f}); download a wider period")
        out[f] = [h[c][k] for c in cols]
    return out


def nyc_weather_features(raw, train_rows):
    """Standardized NYC weather (NYC_WEATHER_NAMES): temperature and wind by the train period's mean / sd,
    precipitation, snowfall and snow depth (cm) as log1p, cloud cover as a fraction."""
    t = raw[train_rows]
    mean, std = t[:, :2].mean(0), t[:, :2].std(0) + 1e-6
    return np.concatenate([(raw[:, :2] - mean) / std, np.log1p(raw[:, 2:4]), np.log1p(100 * raw[:, 4:5]),
                           raw[:, 5:6] / 100], 1).astype(np.float32)


def window(feats, i, t_in, t_out):
    """The window's features, (t_in + t_out, D). Calendar columns are the steps' own (known ahead); every
    weather column of the target steps is the last input step's (future weather isn't known)."""
    w = feats[i:i + t_in + t_out].copy()
    w[t_in:, len(CALENDAR_NAMES):] = w[t_in - 1, len(CALENDAR_NAMES):]
    return w
