import math


def safe_point(x, y, next_center, next_radius, zone_margin):
    radius = max(next_radius - zone_margin, 0.0)
    dx, dy = next_center[0] - x, next_center[1] - y
    distance = math.hypot(dx, dy) or 1.0
    if distance <= radius:
        return (next_center[0], next_center[1])
    return (
        next_center[0] - (dx / distance) * radius,
        next_center[1] - (dy / distance) * radius,
    )


def find_cover(terrain, config, eye_height, x, y, z, threat, next_center, next_radius):
    if terrain is None:
        return None
    tx, ty, tz = threat["pos"]
    bounds = terrain.bounds
    best = None
    best_cost = float("inf")
    for i in range(24):
        angle = i * (2 * math.pi / 24) + 0.2618
        for radius in (25.0, 45.0, 65.0):
            cx = x + radius * math.cos(angle)
            cy = y + radius * math.sin(angle)
            if not (bounds["minX"] < cx < bounds["maxX"] and bounds["minY"] < cy < bounds["maxY"]):
                continue
            if math.hypot(cx - next_center[0], cy - next_center[1]) > next_radius - 10.0:
                continue
            mech_config = config.get("mech", {})
            if not terrain.reachable(
                cx,
                cy,
                z,
                float(mech_config.get("max_step", 1.2)),
                float(mech_config.get("max_slope_deg", 50.0)),
            ):
                continue
            cover_z = terrain.ground(cx, cy) + eye_height
            if not terrain.blocked(tx, ty, tz + eye_height, cx, cy, cover_z):
                continue
            cost = (
                math.hypot(cx - x, cy - y) * 0.6
                + math.hypot(cx - tx, cy - ty) * 0.4
            )
            if cost < best_cost:
                best, best_cost = (cx, cy), cost
    return best


def silent_factor(me, config):
    walk = float(me["stats"].get("walk_speed", 11.7))
    step = float(config.get("sound", {}).get("step_speed", 5.0))
    if walk <= step:
        return 0.95
    return max(0.05, min(0.9, (step * 0.9) / walk))


def pick_upgrade(obs, config):
    offer = obs["offer"]
    if not offer:
        return None
    choices = offer["choices"]
    me = obs["self"]
    hp_fraction = me["hp"] / me["max_hp"]
    held = {weapon["kind"] for weapon in me["weapons"] if weapon}
    definitions = {upgrade["id"]: upgrade for upgrade in config.get("upgrades", [])}
    expires = float(offer.get("expires", 5.0))
    observed_time = float(obs.get("t", 0.0))
    remaining = (
        expires - observed_time
        if expires > observed_time
        else expires
    )

    best_index, best_score = 0, float("-inf")
    for i, choice in enumerate(choices):
        choice_id = choice.get("id", "")
        text = (choice.get("name", "") + " " + choice.get("desc", "")).lower()
        definition = definitions.get(choice_id, {})
        if definition.get("requires") and definition["requires"] not in held:
            continue

        score = 0.0
        if choice_id in ("armor", "repair", "regen") or any(
            token in text for token in ("armor", "repair", "health", " hp")
        ):
            score += 100.0 - hp_fraction * 45.0
            if choice_id == "armor" and hp_fraction < 0.6:
                score += 35.0
        if "sensor" in choice_id or "sensor" in text:
            score += 28.0
        if any(token in choice_id for token in ("speed", "boost", "mobility", "walk")) or "speed" in text:
            score += 24.0
        if any(token in choice_id for token in ("damage", "ammo", "fire", "rate", "reload")) or any(
            token in text for token in ("damage", "fire rate", "reload")
        ):
            score += 42.0
        if choice_id.startswith(("gun_", "mortar_", "laser_", "lightning_")):
            kind = choice_id.split("_", 1)[0]
            score -= 55.0 if kind not in held else 38.0
        if any(token in choice_id for token in ("cube", "size")):
            score += 12.0
        if "cube" in choice_id or "cube" in text:
            score += 20.0 + 4.0 * me["cubes"]
        for kind in ("gun", "mortar", "laser", "lightning"):
            if kind in choice_id or kind in text:
                score += 35.0 if kind in held else -30.0
        if remaining < 1.5 and choice_id in ("armor", "repair", "regen"):
            score += 15.0
        if score > best_score:
            best_index, best_score = i, score
    return best_index


def update_tracking(tracked, obs, memory_seconds):
    tick = int(obs["tick"])
    for actor in obs["mechs"]:
        if actor["kind"] != "mech":
            continue
        tracked[actor["id"]] = {
            "pos": list(actor["pos"]),
            "vel": list(actor.get("vel", [0.0, 0.0, 0.0])),
            "aim": list(actor.get("aim", [0.0, 0.0])),
            "hp": float(actor.get("hp", 1400)),
            "weapons": list(actor.get("weapons", [])),
            "charging": bool(actor.get("charging", False)),
            "seen": tick,
        }
    cutoff = tick - int(memory_seconds * 20)
    return {
        actor_id: target
        for actor_id, target in tracked.items()
        if target["seen"] >= cutoff
    }


def select_target(me, obs, tracked):
    x, y, _ = me["pos"]
    hp_fraction = me["hp"] / me["max_hp"]
    best = None
    best_score = float("-inf")
    for actor_id, target in tracked.items():
        tx, ty, _ = target["pos"]
        distance = math.hypot(tx - x, ty - y)
        if distance > 1000:
            continue
        target_hp = target["hp"] / 1400.0
        score = 90.0 - distance * 0.45
        score += max(0.0, 0.8 - target_hp) * 55.0
        score -= hp_fraction * 8.0
        if target["charging"]:
            score -= 25.0
            score += 18.0 if distance < 70 else -10.0
        if "lightning" in target["weapons"]:
            score -= 8.0 if distance < 45 else 5.0
        if "mortar" in target["weapons"]:
            score += 12.0 if distance > 60 else -12.0
        if "laser" in target["weapons"]:
            score += 8.0 if distance > 70 else -8.0
        if target["seen"] == int(obs["tick"]):
            score += 15.0
        if distance < 12:
            score -= 18.0
        if distance > 220:
            score -= (distance - 220) * 0.12
        if score > best_score:
            best_score = score
            best = {
                "pos": list(target["pos"]),
                "vel": list(target["vel"]),
                "weapons": list(target["weapons"]),
                "charging": target["charging"],
                "hp": target["hp"],
                "id": actor_id,
            }
    return best


def movement_direction(me, target, safe_target, obs):
    x, y, _ = me["pos"]
    if target is None:
        return None
    tx, ty, _ = target["pos"]
    distance = math.hypot(tx - x, ty - y) or 1.0
    toward = [(tx - x) / distance, (ty - y) / distance]
    away = [-toward[0], -toward[1]]

    weapons = {
        me["weapons"][i]["kind"]
        for i in range(len(me["weapons"]))
        if me["weapons"][i]
    }
    ideal = 35.0
    if "lightning" in weapons:
        ideal = 32.0
    elif "laser" in weapons:
        ideal = 95.0
    elif "mortar" in weapons:
        ideal = 120.0
    elif "gun" in weapons:
        ideal = 70.0

    target_weapons = set(target.get("weapons", []))
    hp_fraction = me["hp"] / me["max_hp"]
    if hp_fraction < 0.35 or (target.get("charging") and distance < 140):
        vx, vy = away
    elif distance < ideal * 0.55:
        vx, vy = away
    elif distance > ideal * 1.35:
        vx, vy = toward
    else:
        vx, vy = -toward[1], toward[0]
        if int(obs["tick"] / 20) % 2:
            vx, vy = vy, vx
            vx, vy = -vx, -vy

    if "mortar" in target_weapons and distance < 65:
        vx, vy = away
    if not me["in_zone"]:
        zx, zy = safe_target
        vx, vy = zx - x, zy - y

    length = math.hypot(vx, vy) or 1.0
    return [vx / length, vy / length]


def choose_state(me, obs, target):
    hp_fraction = me["hp"] / me["max_hp"]
    if not me["in_zone"]:
        return "out_of_zone"
    if target is not None:
        target_distance = math.hypot(
            target["pos"][0] - me["pos"][0],
            target["pos"][1] - me["pos"][1],
        )
        if hp_fraction < 0.32 or target_distance < 18:
            return "disengage"
        return "engage"
    if obs["cubes"]:
        return "collect"
    return "zone"
