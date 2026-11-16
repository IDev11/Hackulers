"""Temporal features from observed tracking within each half."""

import numpy as np


def blocks_for(frame, period, clock, within, fps):
    """Reset temporal history at half boundaries and any timeline discontinuity."""
    active = within & np.isfinite(clock) & np.isin(period, [1, 2])
    continuous = np.zeros(len(frame), dtype=bool)
    continuous[1:] = (active[1:] & active[:-1]
                      & (period[1:] == period[:-1])
                      & (frame[1:] == frame[:-1] + 1)
                      & np.isclose(np.diff(clock), 1.0 / fps, atol=1e-5, rtol=0))
    starts = np.flatnonzero(active & ~continuous)
    stops = np.flatnonzero(active & ~np.r_[continuous[1:], False]) + 1
    if len(starts) != len(stops):
        raise ValueError("Unbalanced temporal blocks")
    return list(zip(starts.tolist(), stops.tolist())), continuous


def rolling_mean(signal, width, blocks):
    """A trailing complete window; any unknown input makes the result unknown."""
    result = np.full(len(signal), np.nan, dtype=np.float64)
    for start, stop in blocks:
        if stop - start < width:
            continue
        x = signal[start:stop]
        finite = np.isfinite(x)
        sums = np.r_[0.0, np.cumsum(np.where(finite, x, 0.0), dtype=np.float64)]
        counts = np.r_[0, np.cumsum(finite, dtype=np.int64)]
        good = counts[width:] - counts[:-width] == width
        means = (sums[width:] - sums[:-width]) / width
        result[start + width - 1:stop] = np.where(good, means, np.nan)
    return result


def lag(signal, width, blocks):
    result = np.full(len(signal), np.nan, dtype=np.float64)
    for start, stop in blocks:
        if stop - start > width:
            result[start + width:stop] = signal[start:stop - width]
    return result


def duration(condition, known, blocks, fps):
    """Consecutive known samples / fps; missing input resets the counter."""
    result = np.full(len(condition), np.nan, dtype=np.float64)
    for start, stop in blocks:
        run = 0
        for i in range(start, stop):
            if not known[i]:
                run = 0
                continue
            run = run + 1 if condition[i] else 0
            result[i] = run / fps
    return result


def team_speed_means(common, continuous, minimum, max_speed):
    """Mean of individual speed magnitudes, not velocity of the team centre."""
    xy = common["player_xy_m"].astype(np.float64)
    teams = common["player_team"]
    clock = common["time_seconds"]
    if xy.ndim != 3 or xy.shape[-1] != 2 or teams.shape != (xy.shape[1],):
        raise ValueError("Unexpected common player input shape")
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / np.diff(clock)[:, None]
    valid = np.isfinite(speed) & continuous[:, None] & (speed <= max_speed)
    speed[~valid] = np.nan
    result = {}
    for code, name in [(0, "H"), (1, "A")]:
        group = speed[:, teams == code]
        count = np.isfinite(group).sum(axis=1)
        means = np.full(len(clock), np.nan, dtype=np.float64)
        np.divide(np.nansum(group, axis=1), count, out=means, where=count >= minimum)
        result[name] = means
    return result


def boundary_cues(get, blocks, clock, observed, config):
    """Conservative observed crossings; never assert that an unobserved exit did not occur."""
    n = len(clock)
    margin = np.minimum(get("ball_distance_touchline_m"), get("ball_distance_goal_line_m"))
    speed = get("ball_speed_mps")
    slow_duration = get("ball_slow_duration_s")
    names = ["observed_exit_event", "observed_entry_event", "observed_outside_confirmed",
             "seconds_since_observed_exit", "seconds_since_observed_entry",
             "observed_motion_onset", "seconds_since_observed_motion_onset"]
    result = {name: np.full(n, np.nan, dtype=np.float64) for name in names}
    tolerance = float(config["outside_margin_m"])
    confirmation = int(config["boundary_confirmation_samples"])
    max_age = float(config["max_boundary_crossing_seconds"])
    for start, stop in blocks:
        state = None
        inside_run = outside_run = 0
        inside_anchor = outside_anchor = None
        exit_time = entry_time = motion_time = None
        for i in range(start, stop):
            position_known = observed[i] and np.isfinite(margin[i])
            previous_known = i > start and observed[i - 1] and np.isfinite(margin[i - 1])
            continuous_motion = position_known and previous_known and np.isfinite(speed[i])
            if not position_known or (previous_known and not np.isfinite(speed[i])):
                state = None
                inside_run = outside_run = 0
                inside_anchor = outside_anchor = None
            elif not previous_known:
                state = None
                inside_run = outside_run = 0
                inside_anchor = outside_anchor = None
            if continuous_motion:
                result["observed_exit_event"][i] = 0.0
                result["observed_entry_event"][i] = 0.0
            if position_known:
                if margin[i] >= tolerance:
                    inside_run += 1
                    outside_run = 0
                    inside_anchor = i
                    if inside_run >= confirmation:
                        if (state == "outside" and outside_anchor is not None
                                and clock[i] - clock[outside_anchor] <= max_age
                                and continuous_motion):
                            result["observed_entry_event"][i] = 1.0
                            entry_time = clock[i]
                        state = "inside"
                        result["observed_outside_confirmed"][i] = 0.0
                elif margin[i] <= -tolerance:
                    outside_run += 1
                    inside_run = 0
                    outside_anchor = i
                    if outside_run >= confirmation:
                        if (state == "inside" and inside_anchor is not None
                                and clock[i] - clock[inside_anchor] <= max_age
                                and continuous_motion):
                            result["observed_exit_event"][i] = 1.0
                            exit_time = clock[i]
                        state = "outside"
                        result["observed_outside_confirmed"][i] = 1.0
                else:
                    inside_run = outside_run = 0
            if (continuous_motion and np.isfinite(speed[i - 1])
                    and np.isfinite(slow_duration[i - 1])):
                onset = (slow_duration[i - 1] >= config["minimum_pre_motion_slow_duration_s"]
                         and speed[i] >= config["motion_speed_threshold_mps"])
                result["observed_motion_onset"][i] = float(onset)
                if onset:
                    motion_time = clock[i]
            for key, event_time in [("seconds_since_observed_exit", exit_time),
                                    ("seconds_since_observed_entry", entry_time),
                                    ("seconds_since_observed_motion_onset", motion_time)]:
                if event_time is not None:
                    result[key][i] = clock[i] - event_time
    return result


def build_extra(base_values, base_schema, common, config):
    frame, period, clock, within = (common[k] for k in ["frame_id", "period", "time_seconds", "within_half"])
    n = len(frame)
    fps = float(config["fps"])
    blocks, continuous = blocks_for(frame, period, clock, within, fps)
    index = {feature["name"]: i for i, feature in enumerate(base_schema["features"])}
    if len(index) != base_values.shape[1]:
        raise ValueError("Base schema differs from input values")

    def get(name):
        return base_values[:, index[name]].astype(np.float64)

    columns, features = [], []

    def add(name, values, unit, definition):
        values = np.asarray(values, dtype=np.float64).copy()
        if values.shape != (n,):
            raise ValueError(f"Unexpected shape for {name}")
        values[~within] = np.nan
        values[~np.isfinite(values)] = np.nan
        columns.append(values)
        features.append({"name": name, "unit": unit, "definition": definition})

    windows = [(str(seconds).replace(".", "p"), int(round(float(seconds) * fps)))
               for seconds in config["windows_seconds"]]
    if any(width < 1 or not np.isclose(width / fps, seconds)
           for (_, width), seconds in zip(windows, config["windows_seconds"])):
        raise ValueError("Every window must be an integer number of samples")
    minimum = int(config["min_players_with_velocity_per_team"])
    means = team_speed_means(common, continuous, minimum, config["player_velocity_qc_mps"])
    slow, known, slow_means, slowdown = {}, {}, {}, {}
    for team in ["H", "A"]:
        known[team] = within & (get(f"{team}_players_with_velocity") >= minimum)
        slow[team] = get(f"{team}_slow_players_fraction")
        slow[team][~known[team]] = np.nan
        means[team][~known[team]] = np.nan
        add(f"{team}_speed_mean_mps", means[team], "m/s",
            "Mean magnitude of locally continuous player velocities admitted by QC; minimum support configured")
        for suffix, width in windows:
            mean = rolling_mean(means[team], width, blocks)
            fraction = rolling_mean(slow[team], width, blocks)
            slowing = lag(mean, width, blocks) - mean
            slow_means[team, suffix] = fraction
            slowdown[team, suffix] = slowing
            add(f"{team}_speed_mean_trailing_{suffix}s_mps", mean, "m/s",
                "Trailing mean team speed; requires a complete window of finite supported observations")
            add(f"{team}_slowdown_{suffix}s_mps", slowing, "m/s",
                "Mean speed in the preceding nonoverlapping window minus current window; positive means slowing")
            add(f"{team}_slow_fraction_trailing_{suffix}s", fraction, "fraction",
                "Trailing mean fraction of players classified as slow by the base feature; no game-state label")
            add(f"{team}_velocity_support_fraction_{suffix}s", rolling_mean(known[team].astype(float), width, blocks), "fraction",
                "Fraction of trailing samples with enough player velocities; includes unavailable samples as absent support")
        condition = slow[team] >= config["slow_players_fraction_threshold"]
        add(f"{team}_collective_slow_duration_s", duration(condition, np.isfinite(slow[team]), blocks, fps), "s",
            "Consecutive supported samples with enough slow players; threshold is a cue, not D")
    joint_known = np.isfinite(slow["H"]) & np.isfinite(slow["A"])
    joint_fraction = np.minimum(slow["H"], slow["A"])
    add("both_teams_slow_fraction_min", joint_fraction, "fraction",
        "Minimum slow-player fraction across the two supported teams")
    add("both_teams_collective_slow_duration_s",
        duration(joint_fraction >= config["slow_players_fraction_threshold"], joint_known, blocks, fps), "s",
        "Consecutive samples in which both supported teams have enough slow players")
    for suffix, _ in windows:
        add(f"both_teams_slowdown_min_{suffix}s_mps",
            np.minimum(slowdown["H", suffix], slowdown["A", suffix]), "m/s",
            "Minimum team slowdown; positive only when both team mean speeds decrease")
        add(f"both_teams_speed_increase_min_{suffix}s_mps",
            np.minimum(-slowdown["H", suffix], -slowdown["A", suffix]), "m/s",
            "Minimum team speed increase; positive only when both increase; does not prove a restart")
        add(f"both_teams_slow_fraction_min_trailing_{suffix}s",
            np.minimum(slow_means["H", suffix], slow_means["A", suffix]), "fraction",
            "Minimum trailing mean slow-player fraction across teams")
    observed = common["ball_available"].astype(bool) & np.all(np.isfinite(common["ball_xy_m"]), axis=1)
    for suffix, width in windows:
        add(f"ball_observation_fraction_{suffix}s", rolling_mean(observed.astype(float), width, blocks), "fraction",
            "Fraction of trailing samples containing an admitted ball observation; no hidden ball is read")
    cues = boundary_cues(get, blocks, clock, observed, config)
    descriptions = {
        "observed_exit_event": ("binary", "Confirmed inside-to-outside tracking crossing; unavailable when evidence is missing; not a D label"),
        "observed_entry_event": ("binary", "Confirmed outside-to-inside tracking crossing; ball return alone is not a legal restart"),
        "observed_outside_confirmed": ("binary", "Ball observed outside configured margin for consecutive samples; unknown in the boundary band or without sufficient evidence"),
        "seconds_since_observed_exit": ("s", "Age of last confirmed observed exit; unknown before the first such event, retained through ball missing"),
        "seconds_since_observed_entry": ("s", "Age of last confirmed observed return; does not assert that play restarted"),
        "observed_motion_onset": ("binary", "Observed speed increase after a supported slow-ball run; not a kick or restart detector"),
        "seconds_since_observed_motion_onset": ("s", "Age of last observed motion onset; does not identify a legal restart")
    }
    for name, (unit, definition) in descriptions.items():
        add(name, cues[name], unit, definition)
    age = cues["seconds_since_observed_exit"]
    for seconds in config["exit_recency_seconds"]:
        suffix = str(seconds).replace(".", "p")
        recent = np.where(np.isfinite(age), (age <= seconds).astype(float), np.nan)
        add(f"observed_exit_recent_{suffix}s", recent, "binary",
            "Last confirmed exit was this recent; unknown before any confirmed exit, no claim about unobserved exits")
    opponents = get("teams_nearest_opponents_distance_m")
    close = np.where(np.isfinite(opponents),
                     (opponents <= config["close_opponents_distance_m"]).astype(float), np.nan)
    add("close_opponents_duration_s", duration(close == 1, np.isfinite(close), blocks, fps), "s",
        "Duration of close opposing-player positions; the nearest pair may change, no contact or foul inferred")
    for suffix, width in windows:
        mean = rolling_mean(opponents, width, blocks)
        add(f"opponents_min_distance_change_{suffix}s_m", mean - lag(mean, width, blocks), "m",
            "Change between successive window means of nearest-opponent distance; pairs need not be identical")
        add(f"close_opponents_fraction_{suffix}s", rolling_mean(close, width, blocks), "fraction",
            "Fraction of trailing samples with a close opposing pair; no physical-contact label")
    names = [f["name"] for f in features]
    if len(set(names)) != len(names) or set(names) & set(index):
        raise ValueError("Feature names are duplicated")
    return np.column_stack(columns).astype(np.float32), features
