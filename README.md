# GTA VI launch-day Twitch forecast

[![GTA VI Twitch CCU forecast](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/zaldie/gta6-twitch-forecast-01/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/zaldie/gta6-twitch-forecast-01/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

Forecasts how many people will be watching Grand Theft Auto VI live on Twitch in
each hour of its release day (2026-11-19 UTC), hours 2..24, as five quantiles an
hour — from the first hour's viewer count and the launch-day curves of ~40 past
big game launches.

## The model

A launch day has a strong shape: the first hours dip, the count surges as the
world wakes up, and it peaks in the evening — and how far it climbs depends on
how big the launch already is at hour 1. The entry fits that shape three ways
and blends them:

- a pooled spline-ridge fit of log(viewers(h)/viewers(1)) on the hour and on
  log hour-1 size;
- the same fit with a second size feature, the hour-1 surge over the category's
  own pre-release level;
- a blend of the pooled per-hour median curve and a per-hour regression of the
  log-ratio on log hour-1 size.

Size effects are winsorised at the 90th percentile of past launches, so an
unprecedented hour-1 print cannot extrapolate the evening into nonsense — the
biggest past launches' evenings flatten near 1× their first hour.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, deterministic, no network. One case takes
about 0.2 seconds.

## Layout

- `main.py` — reads each `<id>.in` case, writes `<id>.out` with 23 rows
  (hours 2..24, five quantiles each)

## License

MIT. See `LICENSE`.
