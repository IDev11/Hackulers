"""Base features from the frozen Native tracking extractor."""

import warnings

import numpy as np


def _row_quantile(values, q):
    """Return a linear quantile per row, ignoring missing observations."""
    ordered = np.sort(values, axis=1)
    count = np.isfinite(values).sum(axis=1)
    rank = np.maximum(count - 1, 0) * q
    rows = np.arange(len(values))
    lo = np.floor(rank).astype(int)
    hi = np.ceil(rank).astype(int)
    lower = ordered[rows, lo]
    upper = ordered[rows, hi]
    weight = (rank - lo).astype(values.dtype)

    # Match the two interpolation branches used by the saved extractor.
    diff = np.subtract(upper, lower)
    with np.errstate(invalid="ignore"):
        result = lower + diff * weight
        use_upper = weight >= 0.5
        complement = (1 - (rank - lo)).astype(values.dtype)
        result[use_upper] = upper[use_upper] - diff[use_upper] * complement[use_upper]
    result[count == 0] = np.nan
    return result


PLAYER_SPEED_QC = 20.0
BALL_SPEED_QC = 100.0
SLOW_BALL_SPEED = 0.5


def same_step(frame, period, clock):
    ok = np.zeros(len(frame), dtype=bool)
    ok[1:] = ((period[1:] == period[:-1]) & (period[1:] > 0)
              & (frame[1:] == frame[:-1] + 1)
              & np.isclose(clock[1:] - clock[:-1], 0.1, atol=1e-5, rtol=0))
    return ok


def derivative(xy, frame, period, clock, max_speed):
    dt = np.full(len(frame), np.nan)
    dt[1:] = clock[1:] - clock[:-1]
    shape = (len(frame) - 1,) + (1,) * (xy.ndim - 1)
    v = np.full_like(xy, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        v[1:] = (xy[1:] - xy[:-1]) / dt[1:].reshape(shape)
    valid = np.all(np.isfinite(v), axis=-1)
    step = same_step(frame, period, clock)
    valid &= step.reshape((len(frame),) + (1,) * (valid.ndim - 1))
    speed = np.linalg.norm(v, axis=-1)
    valid &= ~(valid & (speed > max_speed))
    v[~valid] = np.nan
    return v


def make_features(ball, players, player_team, frame, period, clock, length, width):
    """Calculate the 78 base features from observed ball and player coordinates."""
    n = len(frame)
    cols, schema = [], []
    def add(name, values, unit, definition):
        a = np.asarray(values, dtype=np.float32)
        assert a.shape == (n,), (name, a.shape)
        a = a.copy()
        a[~np.isfinite(a)] = np.nan
        cols.append(a)
        schema.append({"name": name, "unit": unit, "definition": definition})
    observed = np.all(np.isfinite(ball), axis=1)
    add("ball_available", observed, "binary", "Finite ball coordinates admitted by the input policy; available in both providers")
    add("ball_x_m", ball[:, 0], "m", "Longitudinal coordinate, pitch centre origin")
    add("ball_y_m", ball[:, 1], "m", "Lateral coordinate, pitch centre origin")
    bv = derivative(ball, frame, period, clock, BALL_SPEED_QC)
    bs = np.linalg.norm(bv, axis=1)
    for j, name in enumerate(["ball_vx_mps", "ball_vy_mps"]):
        add(name, bv[:, j], "m/s", "Backward difference between adjacent observed samples in the same half; missing across gaps")
    add("ball_speed_mps", bs, "m/s", "Euclidean norm of valid ball velocity")
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = bv / bs[:, None]
    add("ball_direction_x", direction[:, 0], "unitless", "Unit x component of valid nonzero ball velocity")
    add("ball_direction_y", direction[:, 1], "unitless", "Unit y component of valid nonzero ball velocity")
    ba = derivative(bv, frame, period, clock, np.inf)
    add("ball_acceleration_mps2", np.linalg.norm(ba, axis=1), "m/s2", "Backward difference of adjacent valid ball velocities")
    slow = np.full(n, np.nan)
    step = same_step(frame, period, clock)
    duration = 0.0
    for i in range(n):
        if not step[i]:
            duration = 0.0
        if np.isfinite(bs[i]):
            duration = duration + .1 if bs[i] <= SLOW_BALL_SPEED else 0.0
            slow[i] = duration
        else:
            duration = 0.0
    add("ball_slow_duration_s", slow, "s", "Consecutive valid velocity samples <=0.5 m/s; not a dead-ball rule")
    add("ball_distance_touchline_m", width / 2 - np.abs(ball[:, 1]), "m", "Signed distance to nearest touchline; negative beyond the line")
    add("ball_distance_goal_line_m", length / 2 - np.abs(ball[:, 0]), "m", "Signed distance to nearest end line; negative beyond the line")
    for sign, name in [(-1, "negative_x"), (1, "positive_x")]:
        add(f"ball_distance_goal_{name}_m", np.linalg.norm(ball - [sign * length / 2, 0], axis=1), "m", "2D distance to a goal centre; no assumption about the team attacking it")
    pv = derivative(players, frame, period, clock, PLAYER_SPEED_QC)
    centres, grids, nearest, counts = {}, {}, {}, {}
    for team, prefix in [(0, "H"), (1, "A")]:
        xy = players[:, player_team == team]
        v = pv[:, player_team == team]
        available = np.all(np.isfinite(xy), axis=2)
        count = available.sum(axis=1)
        counts[prefix] = count
        centre = np.nansum(xy, axis=1) / np.maximum(count[:, None], 1)
        centre[count == 0] = np.nan
        centres[prefix] = centre
        add(f"{prefix}_players_available", count, "count", "Players with both finite coordinates, independent of their array order")
        for axis, ax in enumerate(["x", "y"]):
            add(f"{prefix}_centre_{ax}_m", centre[:, axis], "m", "Mean coordinate of available teammates; accompanied by availability count")
            add(f"{prefix}_extent_{ax}_m", _row_quantile(xy[:, :, axis], .9) - _row_quantile(xy[:, :, axis], .1), "m", "90th minus 10th percentile among available teammates")
        radius = np.sqrt(np.nansum((xy - centre[:, None]) ** 2, axis=(1, 2)) / np.maximum(count, 1))
        radius[count == 0] = np.nan
        add(f"{prefix}_spread_m", radius, "m", "Root mean squared distance of available teammates from their centre")
        centre_v = derivative(centre, frame, period, clock, np.inf)
        for axis, ax in enumerate(["x", "y"]):
            add(f"{prefix}_centre_v{ax}_mps", centre_v[:, axis], "m/s", "Change in observed team centre; availability changes may affect it")
        for axis, ax in enumerate(["x", "y"]):
            extent = _row_quantile(xy[:, :, axis], .9) - _row_quantile(xy[:, :, axis], .1)
            extent2d = np.column_stack([extent, np.zeros(n)])
            ev = derivative(extent2d, frame, period, clock, np.inf)
            add(f"{prefix}_extent_{ax}_change_mps", ev[:, 0], "m/s", "Temporal change in robust team extent")
        speed = np.linalg.norm(v, axis=2)
        nv = np.isfinite(speed).sum(axis=1)
        add(f"{prefix}_players_with_velocity", nv, "count", "Players with a valid locally continuous velocity")
        for q, suffix in [(.25, "p25"), (.5, "median"), (.75, "p75")]:
            add(f"{prefix}_speed_{suffix}_mps", _row_quantile(speed, q), "m/s", "Quantile of valid player speeds")
        stationary = np.sum(np.isfinite(speed) & (speed <= .5), axis=1) / np.maximum(nv, 1)
        stationary[nv == 0] = np.nan
        add(f"{prefix}_slow_players_fraction", stationary, "fraction", "Fraction of players with valid speeds <=0.5 m/s; not a game-state rule")
        xe, ye = np.linspace(-length / 2, length / 2, 4), np.linspace(-width / 2, width / 2, 3)
        grid = []
        for xi in range(3):
            for yi in range(2):
                inside = available & (xy[:, :, 0] >= xe[xi]) & (xy[:, :, 1] >= ye[yi])
                inside &= (xy[:, :, 0] <= xe[xi + 1]) if xi == 2 else (xy[:, :, 0] < xe[xi + 1])
                inside &= (xy[:, :, 1] <= ye[yi + 1]) if yi == 1 else (xy[:, :, 1] < ye[yi + 1])
                fraction = inside.sum(axis=1) / np.maximum(count, 1)
                fraction[count == 0] = np.nan
                grid.append(fraction)
                add(f"{prefix}_grid_{xi}_{yi}_fraction", fraction, "fraction", "Fraction of available teammates in a fixed 3x2 pitch cell; no coordinate clipping")
        grids[prefix] = np.column_stack(grid)
        distances = np.linalg.norm(xy - ball[:, None], axis=2)
        ordered = np.sort(distances, axis=1)
        nearest[prefix] = ordered[:, 0]
        for k in range(3):
            add(f"{prefix}_ball_distance_rank{k + 1}_m", ordered[:, k], "m", "Sorted ball distance among available teammates; missing without observed ball")
        for radius_m in [2, 5, 10]:
            close = np.sum(np.isfinite(distances) & (distances <= radius_m), axis=1).astype(float)
            close[~observed | (count == 0)] = np.nan
            add(f"{prefix}_players_within_{radius_m}m_ball", close, "count", "Observed teammates within radius of observed ball; unavailable when ball or team is absent")
    delta = centres["A"] - centres["H"]
    add("teams_centre_dx_m", delta[:, 0], "m", "A centre x minus H centre x")
    add("teams_centre_dy_m", delta[:, 1], "m", "A centre y minus H centre y")
    add("teams_centre_distance_m", np.linalg.norm(delta, axis=1), "m", "Distance between observed team centres")
    add("teams_grid_overlap", np.sum(np.minimum(grids["H"], grids["A"]), axis=1), "fraction", "Sum of cellwise minima of observed team distributions")
    add("ball_nearest_distance_A_minus_H_m", nearest["A"] - nearest["H"], "m", "Difference in nearest-team distance; positive means H is closer")
    min_opponent = np.full(n, np.nan)
    hxy, axy = players[:, player_team == 0], players[:, player_team == 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for start in range(0, n, 1024):
            stop = min(start + 1024, n)
            dist = np.linalg.norm(hxy[start:stop, :, None] - axy[start:stop, None], axis=3)
            min_opponent[start:stop] = np.nanmin(dist, axis=(1, 2))
    add("teams_nearest_opponents_distance_m", min_opponent, "m", "Minimum observed H-A distance; no stable identity used as a feature")
    last_xy = np.full_like(ball, np.nan)
    age = np.full(n, np.nan)
    previous_period, last_time, last_position = 0, np.nan, None
    missing_duration = np.full(n, np.nan)
    run = 0.0
    for i in range(n):
        if period[i] == 0 or period[i] != previous_period:
            last_time, last_position, run = np.nan, None, 0.0
        previous_period = period[i]
        if period[i] == 0:
            continue
        if observed[i]:
            last_time, last_position, run = clock[i], ball[i].copy(), 0.0
        else:
            run = run + .1 if step[i] else .1
        missing_duration[i] = run
        if last_position is not None:
            last_xy[i], age[i] = last_position, clock[i] - last_time
    add("ball_last_observed_x_m", last_xy[:, 0], "m", "Most recent admitted observation within this half, explicitly stale when age>0")
    add("ball_last_observed_y_m", last_xy[:, 1], "m", "Most recent admitted observation within this half")
    add("ball_age_last_observation_s", age, "s", "Elapsed match time since last admitted observation; missing before first observation")
    add("ball_missing_run_s", missing_duration, "s", "Current consecutive missing duration; resets on observation and at half boundary")
    x = np.column_stack(cols).astype(np.float32)
    names = [s["name"] for s in schema]
    assert len(names) == len(set(names))
    assert not any(any(word in name for word in ["phase", "possession", "player_id", "match_id", "score", "role", "is_detected"]) for name in names)
    return x, schema
