import math


def led_aim(me, target, config, eye_height, weapon=None):
    tx, ty, tz = target["pos"]
    x, y, z = me["pos"]
    tvx, tvy, tvz = target.get("vel", [0.0, 0.0, 0.0])
    dx, dy = tx - x, ty - y
    distance = math.hypot(dx, dy)
    speed = float(weapon.get("stats", {}).get("speed", 0.0)) if weapon else 0.0
    if speed > 0:
        flight = distance / speed
        px, py, pz = tx + tvx * flight, ty + tvy * flight, tz + tvz * flight
    else:
        px, py, pz = tx, ty, tz

    pdx, pdy = px - x, py - y
    predicted_distance = math.hypot(pdx, pdy)
    muzzle = z + float(me["stats"].get("muzzle_height", 7.0))
    if weapon and weapon["kind"] == "mortar":
        gravity = float(config.get("weapons", {}).get("mortar", {}).get("gravity", 22.0))
        shell_speed = float(weapon["stats"].get("speed", 72.0))
        flight = predicted_distance / max(shell_speed, 1.0)
        pitch = math.atan2(
            pz + 4.5 - muzzle + 0.5 * gravity * flight * flight,
            predicted_distance,
        )
    else:
        pitch = math.atan2(pz + 4.5 - muzzle, predicted_distance)
    return [math.atan2(pdy, pdx), max(-0.9, min(0.9, pitch))]


def take_weapon(me, obs, config):
    if not obs.get("pickups"):
        return {}
    radius = float(config.get("loot", {}).get("pickup_radius", 4.0))
    held = {weapon["kind"]: i for i, weapon in enumerate(me["weapons"]) if weapon}
    priority = {"laser": 40, "lightning": 35, "mortar": 32, "gun": 10}
    candidates = []
    for pickup in obs["pickups"]:
        px, py, pz = pickup["pos"]
        distance = math.hypot(px - me["pos"][0], py - me["pos"][1])
        if distance <= radius and abs(pz - me["pos"][2]) <= 4.0:
            kind = pickup["kind"]
            if kind not in held:
                candidates.append((priority.get(kind, 5) - distance * 0.3, pickup))
    if not candidates:
        return {}

    _, pickup = max(candidates, key=lambda pair: pair[0])
    action = {"take": pickup["id"]}
    empty = next((i for i, weapon in enumerate(me["weapons"]) if weapon is None), None)
    if empty is not None:
        action.pop("take", None)
        action["take"] = pickup["id"]
    else:
        kind_priority = {"gun": 10, "mortar": 32, "lightning": 35, "laser": 40}
        drop_index = None
        drop_score = float("inf")
        for i, weapon in enumerate(me["weapons"]):
            if not weapon:
                continue
            if weapon["kind"] == "laser" and float(weapon.get("charge", 0)) > 0:
                continue
            score = kind_priority.get(weapon["kind"], 5)
            if score < drop_score:
                drop_index, drop_score = i, score
        if drop_index is None or kind_priority.get(pickup["kind"], 5) <= drop_score:
            action.pop("take")
        else:
            action["drop"] = drop_index
    return action


def combat_fire(me, target, terrain, config, eye_height):
    fire = [False, False]
    reload = False
    aim = None
    if not target:
        return aim, fire, reload

    x, y, z = me["pos"]
    distance = math.dist(me["pos"], target["pos"])
    blocked = False
    if terrain is not None:
        blocked = terrain.blocked(
            x,
            y,
            z + float(me["stats"].get("eye_height", eye_height)),
            target["pos"][0],
            target["pos"][1],
            target["pos"][2] + 4.5,
        )

    for i, weapon in enumerate(me["weapons"][:2]):
        if not weapon:
            continue
        kind = weapon["kind"]
        stats = weapon["stats"]
        cooldown = float(weapon.get("cooldown", 0.0))

        if kind == "gun":
            ammo = float(weapon.get("ammo", 0))
            reload_left = float(weapon.get("reload", 0))
            mag = float(stats.get("mag", 12))
            if ammo <= 0 or (ammo <= mag * 0.25 and reload_left <= 0):
                reload = reload_left <= 0
            if not blocked and cooldown <= 0.05 and ammo > 0 and distance <= float(stats.get("range", 230)):
                fire[i] = True
            aim = aim or led_aim(me, target, config, eye_height, weapon)
        elif kind == "mortar":
            if cooldown <= 0.05 and distance <= float(stats.get("range", 235)):
                fire[i] = True
            aim = aim or led_aim(me, target, config, eye_height, weapon)
        elif kind == "laser":
            charge_full = float(config.get("weapons", {}).get("laser", {}).get("charge", 0.75))
            min_charge = float(config.get("weapons", {}).get("laser", {}).get("min_charge", 0.15))
            charge = float(weapon.get("charge", 0))
            if not blocked and cooldown <= 0.05 and distance <= float(stats.get("range", 280)):
                fire[i] = not (charge >= charge_full or (charge >= min_charge and distance < 90))
            aim = aim or led_aim(me, target, config, eye_height, weapon)
        elif kind == "lightning":
            energy_cost = float(config.get("weapons", {}).get("lightning", {}).get("energy", 3))
            if (not blocked and cooldown <= 0.05 and me["energy"] >= energy_cost
                    and distance <= float(stats.get("range", 45))):
                fire[i] = True
            aim = aim or led_aim(me, target, config, eye_height, weapon)
    return aim, fire, reload
