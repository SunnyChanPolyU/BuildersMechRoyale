import json
import math
import os
import struct

import mechfight as mf


# ------------------------------------------------------------------ map model

class TerrainMap:
    """Ground heights + sight-blocking prisms. Loaded from hello.map.dir."""

    def __init__(self, map_dir: str, map_files):
        paths = {os.path.basename(f): os.path.join(map_dir, f)
                 for f in map_files}
        wpath = paths.get("world.json")
        tpath = paths.get("terrain.i16")
        if wpath is None or tpath is None:
            raise OSError(f"map manifest lacks expected files: {map_files}")

        with open(wpath, encoding="utf-8") as f:
            w = json.load(f)

        t = w["terrain"]
        self.tx, self.ty = t["minX"], t["minY"]
        self.tstep = t["step"]
        self.tnx, self.tny = t["nx"], t["ny"]
        self.scale = t.get("scale", 0.01)
        with open(tpath, "rb") as f:
            raw = f.read()
        need = 2 * self.tnx * self.tny
        if len(raw) < need:
            raise OSError(f"terrain file short: {len(raw)} < {need}")
        self.h = struct.unpack("<%dh" % (self.tnx * self.tny), raw[:need])

        self.bounds = w["bounds"]
        # Only non-bridge prisms block sight: you see and shoot under bridges.
        self.prisms = []
        for p in w.get("prisms", []):
            ring = p["ring"]
            xs, ys = ring[0::2], ring[1::2]
            holes_flat = p.get("holes", [])
            holes = []
            if holes_flat and isinstance(holes_flat[0], (int, float)):
                holes_flat = [holes_flat]
            for h in holes_flat:
                hx, hy = h[0::2], h[1::2]
                holes.append(list(zip(hx, hy)))
            if p.get("kind") == "bridge":
                continue
            self.prisms.append((p["base"], p["top"],
                                (min(xs), min(ys), max(xs), max(ys)),
                                list(zip(xs, ys)), holes))

    def ground(self, x: float, y: float) -> float:
        """Bilinearly interpolated ground height in meters."""
        fx = (x - self.tx) / self.tstep
        fy = (y - self.ty) / self.tstep
        i0 = max(0, min(self.tnx - 2, int(fx)))
        j0 = max(0, min(self.tny - 2, int(fy)))
        ax = min(max(fx - i0, 0.0), 1.0)
        ay = min(max(fy - j0, 0.0), 1.0)
        g = lambda i, j: self.h[j * self.tnx + i] * self.scale
        return ((g(i0, j0) * (1 - ax) + g(i0 + 1, j0) * ax) * (1 - ay)
                + (g(i0, j0 + 1) * (1 - ax) + g(i0 + 1, j0 + 1) * ax) * ay)

    @staticmethod
    def _in_poly(px, py, poly):
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > py) != (yj > py) and \
               px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi:
                inside = not inside
            j = i
        return inside

    def blocked(self, x1, y1, z1, x2, y2, z2, step=4.0):
        """Any prism wall crossing the segment (x1,y1,z1)->(x2,y2,z2)?"""
        d = math.hypot(x2 - x1, y2 - y1)
        if d < 1e-6:
            return False
        n = max(2, int(d / step))
        for k in range(1, n):
            u = k / n
            px = x1 + (x2 - x1) * u
            py = y1 + (y2 - y1) * u
            pz = z1 + (z2 - z1) * u
            for base, top, bb, ring, holes in self.prisms:
                if not (bb[0] <= px <= bb[2] and bb[1] <= py <= bb[3]):
                    continue
                if top <= pz or base >= pz:
                    continue
                if self._in_poly(px, py, ring) and not any(
                        self._in_poly(px, py, hole) for hole in holes):
                    return True
        return False

    def solid_at(self, x, y, feet_z, height=9.0):
        """True if a point is inside solid building/stadium volume."""
        for base, top, bb, ring, holes in self.prisms:
            if not (bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3]):
                continue
            if not self._in_poly(x, y, ring):
                continue
            if any(self._in_poly(x, y, hole) for hole in holes):
                continue
            # On a roof, feet_z at/above top are outside the solid.
            if base < feet_z + height and top > feet_z:
                return True
        return False

    def reachable(self, x, y, from_z, max_step=1.2, max_slope_deg=50.0):
        """Conservative local reachability check for cover/loot points."""
        b = self.bounds
        if not (b["minX"] < x < b["maxX"] and b["minY"] < y < b["maxY"]):
            return False
        ground = self.ground(x, y)
        if ground + max_step < from_z - 0.5:
            return False
        if ground > from_z + max_step:
            return False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            other = self.ground(x + dx, y + dy)
            slope = math.degrees(math.atan2(abs(other - ground), 1.0))
            if slope > max_slope_deg:
                return False
        return not self.solid_at(x, y, ground)


# -------------------------------------------------------------------- the bot

class Bot:
    name = "ver-4-bot"

    PANIC_DIST = 100.0   # flee if a mech is visible closer than this
    FIGHT_DIST = 60.0    # engage when pressed closer than this
    CUBE_GRAB = 25.0     # grab cubes within this range while calm
    ZONE_MARGIN = 40.0   # preferred inset from the next circle's edge
    COVER_CHECK = 40     # re-plan cover every N ticks (~2 s)

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.eye = float(cfg["mech"].get("eye_height", 8.0))
        self.cfg = cfg
        self.rear_bearing = None
        self.rear_bearing_age = 0
        self.cover_goal = None
        self.next_cover_scan = 0
        self.tracked_mechs = {}
        self.last_damage_tick = -1000
        self.last_hit_tick = -1000
        self.previous_hp = None
        self.last_state = "explore"
        self.match_seed = int(hello.get("seed", 0))
        self.map = None
        try:
            self.map = TerrainMap(hello["map"]["dir"],
                                  hello["map"].get("files", []))
            mf.log(f"ver-4-bot: map loaded, {len(self.map.prisms)} prisms")
        except Exception as e:  # any failure: degrade, never crash
            mf.log(f"ver-4-bot: no map ({e!r}); fallback navigation mode")
        self.cover_goal = None
        self.next_cover_scan = 0

    # ---- goal points ------------------------------------------------------

    def _safe_point(self, x, y, nxt, nr):
        """Interior point of the next circle; stops slightly inside the edge
        when far outside, instead of aiming at the bare centre."""
        r = max(nr - self.ZONE_MARGIN, 0.0)
        dx, dy = nxt[0] - x, nxt[1] - y
        d = math.hypot(dx, dy) or 1.0
        if d <= r:
            return (nxt[0], nxt[1])
        return (nxt[0] - (dx / d) * r, nxt[1] - (dy / d) * r)

    def _find_cover(self, x, y, z, threat, nxt, nr):
        """Nearest spot where a building blocks LOS from `threat`, still
        inside the next circle. Returns (x, y) or None."""
        if self.map is None:
            return None
        tx, ty, tz = threat["pos"]
        b = self.map.bounds
        best, best_cost = None, float("inf")
        for i in range(24):
            ang = i * (2 * math.pi / 24) + 0.2618  # 15 deg offset for variety
            for r in (25.0, 45.0, 65.0):
                cx = x + r * math.cos(ang)
                cy = y + r * math.sin(ang)
                if not (b["minX"] < cx < b["maxX"] and b["minY"] < cy < b["maxY"]):
                    continue
                if math.hypot(cx - nxt[0], cy - nxt[1]) > nr - 10.0:
                    continue          # don't hide into the closing storm
                mech_cfg = self.cfg.get("mech", {})
                if not self.map.reachable(
                        cx, cy, z,
                        float(mech_cfg.get("max_step", 1.2)),
                        float(mech_cfg.get("max_slope_deg", 50.0))):
                    continue
                cz = self.map.ground(cx, cy) + self.eye
                if not self.map.blocked(tx, ty, tz + self.eye, cx, cy, cz):
                    continue
                cost = (math.hypot(cx - x, cy - y) * 0.6
                        + math.hypot(cx - tx, cy - ty) * 0.4)
                if cost < best_cost:
                    best, best_cost = (cx, cy), cost
        return best

    def _silent_factor(self, me):
        walk = float(me["stats"].get("walk_speed", 11.7))
        step = float(self.cfg.get("sound", {}).get("step_speed", 5.0))
        if walk <= step:
            return 0.95
        return max(0.05, min(0.9, (step * 0.9) / walk))

    def _led_aim(self, me, target, weapon=None):
        tx, ty, tz = target["pos"]
        x, y, z = me["pos"]
        tvx, tvy, tvz = target.get("vel", [0.0, 0.0, 0.0])
        dx, dy = tx - x, ty - y
        d = math.hypot(dx, dy)
        speed = float(weapon.get("stats", {}).get("speed", 0.0)) if weapon else 0.0
        if speed > 0:
            flight = d / speed
            px, py, pz = tx + tvx * flight, ty + tvy * flight, tz + tvz * flight
        else:
            px, py, pz = tx, ty, tz
        pdx, pdy = px - x, py - y
        pd = math.hypot(pdx, pdy)
        muzzle = z + float(me["stats"].get("muzzle_height", 7.0))
        if weapon and weapon["kind"] == "mortar":
            gravity = float(self.cfg.get("weapons", {}).get("mortar", {}).get("gravity", 22.0))
            shell_speed = float(weapon["stats"].get("speed", 72.0))
            flight = pd / max(shell_speed, 1.0)
            pitch = math.atan2(pz + 4.5 - muzzle + 0.5 * gravity * flight * flight, pd)
        else:
            pitch = math.atan2(pz + 4.5 - muzzle, pd)
        return [math.atan2(pdy, pdx), max(-0.9, min(0.9, pitch))]

    def _pick_upgrade(self, obs):
        offer = obs["offer"]
        if not offer:
            return None
        choices = offer["choices"]
        me = obs["self"]
        hp_frac = me["hp"] / me["max_hp"]
        held = {w["kind"] for w in me["weapons"] if w}
        defs = {u["id"]: u for u in self.cfg.get("upgrades", [])}
        expires = float(offer.get("expires", 5.0))
        remaining = expires - float(obs.get("t", 0.0)) if expires > float(obs.get("t", 0.0)) else expires
        best_i, best_score = 0, float("-inf")
        for i, choice in enumerate(choices):
            cid = choice.get("id", "")
            text = (choice.get("name", "") + " " + choice.get("desc", "")).lower()
            definition = defs.get(cid, {})
            if definition.get("requires") and definition["requires"] not in held:
                continue
            score = 0.0
            if cid in ("armor", "repair", "regen") or any(s in text for s in ("armor", "repair", "health", " hp")):
                score += 100.0 - hp_frac * 45.0
                if cid == "armor" and hp_frac < 0.6:
                    score += 35.0
            if "sensor" in cid or "sensor" in text:
                score += 28.0
            if any(s in cid for s in ("speed", "boost", "mobility", "walk")) or "speed" in text:
                score += 24.0
            if any(s in cid for s in ("damage", "ammo", "fire", "rate", "reload")) or any(s in text for s in ("damage", "fire rate", "reload")):
                score += 42.0
            if cid.startswith("gun_") or cid.startswith("mortar_") or cid.startswith("laser_") or cid.startswith("lightning_"):
                kind = cid.split("_", 1)[0]
                if kind not in held:
                    score -= 55.0
                else:
                    score += 38.0
            if any(s in cid for s in ("cube", "size")):
                score += 12.0
            if "cube" in cid or "cube" in text:
                score += 20.0 + 4.0 * me["cubes"]
            for kind in ("gun", "mortar", "laser", "lightning"):
                if kind in cid or kind in text:
                    if kind in held:
                        score += 35.0
                    else:
                        score -= 30.0
            if remaining < 1.5 and cid in ("armor", "repair", "regen"):
                score += 15.0
            if score > best_score:
                best_i, best_score = i, score
        return best_i

    def _take_weapon(self, me, obs):
        if not obs.get("pickups"):
            return {}
        radius = float(self.cfg.get("loot", {}).get("pickup_radius", 4.0))
        held = {w["kind"]: i for i, w in enumerate(me["weapons"]) if w}
        priority = {"laser": 40, "lightning": 35, "mortar": 32, "gun": 10}
        candidates = []
        for pickup in obs["pickups"]:
            px, py, pz = pickup["pos"]
            d = math.hypot(px - me["pos"][0], py - me["pos"][1])
            if d <= radius and abs(pz - me["pos"][2]) <= 4.0:
                kind = pickup["kind"]
                if kind not in held:
                    candidates.append((priority.get(kind, 5) - d * 0.3, pickup))
        if not candidates:
            return {}
        _, pickup = max(candidates, key=lambda pair: pair[0])
        action = {"take": pickup["id"]}
        empty = next((i for i, w in enumerate(me["weapons"]) if w is None), None)
        if empty is None:
            kind_priority = {"gun": 10, "mortar": 32, "lightning": 35, "laser": 40}
            drop_i = None
            drop_score = float("inf")
            for i, w in enumerate(me["weapons"]):
                if not w:
                    continue
                if w["kind"] == "laser" and float(w.get("charge", 0)) > 0:
                    continue
                score = kind_priority.get(w["kind"], 5)
                if score < drop_score:
                    drop_i, drop_score = i, score
            if drop_i is None or kind_priority.get(pickup["kind"], 5) <= drop_score:
                action.pop("take")
            else:
                action["drop"] = drop_i
        return action

    def _combat_fire(self, me, target):
        fire = [False, False]
        reload = False
        aim = None
        if not target:
            return aim, fire, reload
        x, y, z = me["pos"]
        d = math.dist(me["pos"], target["pos"])
        blocked = False
        if self.map is not None:
            blocked = self.map.blocked(
                x, y, z + float(me["stats"].get("eye_height", self.eye)),
                target["pos"][0], target["pos"][1],
                target["pos"][2] + 4.5)
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
                if not blocked and cooldown <= 0.05 and ammo > 0 and d <= float(stats.get("range", 230)):
                    fire[i] = True
                aim = aim or self._led_aim(me, target, weapon)
            elif kind == "mortar":
                if cooldown <= 0.05 and d <= float(stats.get("range", 235)):
                    fire[i] = True
                aim = aim or self._led_aim(me, target, weapon)
            elif kind == "laser":
                charge_full = float(self.cfg.get("weapons", {}).get("laser", {}).get("charge", 0.75))
                min_charge = float(self.cfg.get("weapons", {}).get("laser", {}).get("min_charge", 0.15))
                charge = float(weapon.get("charge", 0))
                if not blocked and cooldown <= 0.05 and d <= float(stats.get("range", 280)):
                    fire[i] = not (charge >= charge_full or (charge >= min_charge and d < 90))
                aim = aim or self._led_aim(me, target, weapon)
            elif kind == "lightning":
                energy_cost = float(self.cfg.get("weapons", {}).get("lightning", {}).get("energy", 3))
                if (not blocked and cooldown <= 0.05 and me["energy"] >= energy_cost
                        and d <= float(stats.get("range", 45))):
                    fire[i] = True
                aim = aim or self._led_aim(me, target, weapon)
        return aim, fire, reload

    def _update_tracking(self, obs):
        tick = int(obs["tick"])
        memory_s = float(self.cfg.get("vision", {}).get("memory_s", 5.0))
        for actor in obs["mechs"]:
            if actor["kind"] != "mech":
                continue
            self.tracked_mechs[actor["id"]] = {
                "pos": list(actor["pos"]),
                "vel": list(actor.get("vel", [0.0, 0.0, 0.0])),
                "aim": list(actor.get("aim", [0.0, 0.0])),
                "hp": float(actor.get("hp", 1400)),
                "weapons": list(actor.get("weapons", [])),
                "charging": bool(actor.get("charging", False)),
                "seen": tick,
            }
        cutoff = tick - int(memory_s * 20)
        self.tracked_mechs = {
            actor_id: target for actor_id, target in self.tracked_mechs.items()
            if target["seen"] >= cutoff
        }

    def _select_target(self, me, obs):
        x, y, z = me["pos"]
        hp_frac = me["hp"] / me["max_hp"]
        best = None
        best_score = float("-inf")
        for actor_id, target in self.tracked_mechs.items():
            tx, ty, tz = target["pos"]
            d = math.hypot(tx - x, ty - y)
            if d > 1000:
                continue
            target_hp = target["hp"] / 1400.0
            score = 90.0 - d * 0.45
            score += max(0.0, 0.8 - target_hp) * 55.0
            score -= hp_frac * 8.0
            if target["charging"]:
                score -= 25.0
                score += 18.0 if d < 70 else -10.0
            if "lightning" in target["weapons"]:
                score -= 8.0 if d < 45 else 5.0
            if "mortar" in target["weapons"]:
                score += 12.0 if d > 60 else -12.0
            if "laser" in target["weapons"]:
                score += 8.0 if d > 70 else -8.0
            if target["seen"] == int(obs["tick"]):
                score += 15.0
            if d < 12:
                score -= 18.0
            if d > 220:
                score -= (d - 220) * 0.12
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

    def _best_movement_direction(self, me, target, nxt, nr, obs):
        x, y, z = me["pos"]
        if target is None:
            return None
        tx, ty, _ = target["pos"]
        d = math.hypot(tx - x, ty - y) or 1.0
        toward = [(tx - x) / d, (ty - y) / d]
        away = [-toward[0], -toward[1]]
        weapons = set(me["weapons"][i]["kind"] for i in range(len(me["weapons"])) if me["weapons"][i])
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
        hp_frac = me["hp"] / me["max_hp"]
        if hp_frac < 0.35 or (target.get("charging") and d < 140):
            vx, vy = away
        elif d < ideal * 0.55:
            vx, vy = away
        elif d > ideal * 1.35:
            vx, vy = toward
        else:
            vx, vy = -toward[1], toward[0]
            if int(obs["tick"] / 20) % 2:
                vx, vy = vy, vx
                vx, vy = -vx, -vy

        if "mortar" in target_weapons and d < 65:
            vx, vy = away
        if not me["in_zone"]:
            zx, zy = nxt
            vx, vy = zx - x, zy - y

        n = math.hypot(vx, vy) or 1.0
        return [vx / n, vy / n]

    # ---- one decision -----------------------------------------------------

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        hp_frac = me["hp"] / me["max_hp"]
        act: mf.Action = {"tick": obs["tick"]}

        pick = self._pick_upgrade(obs)
        if pick is not None:
            act["pick"] = pick

        if self.previous_hp is not None and me["hp"] < self.previous_hp:
            self.last_damage_tick = int(obs["tick"])
        for event in reversed(obs.get("events", [])):
            if event.get("type") == "hit":
                self.last_hit_tick = int(obs["tick"])
                break
        self.previous_hp = me["hp"]
        self._update_tracking(obs)

        zone = obs["zone"]
        zc = zone["center"]
        nxt, nr = zone["next_center"], zone["next_radius"]
        tgt = self._safe_point(x, y, nxt, nr)

        target = self._select_target(me, obs)
        threats = []
        for tracked in self.tracked_mechs.values():
            d = math.hypot(tracked["pos"][0] - x, tracked["pos"][1] - y)
            if d < self.PANIC_DIST:
                threats.append((d, tracked))

        flee_yaw = None
        for event in reversed(obs["events"]):
            if event.get("type") == "damaged" and event.get("weapon") != "zone":
                if event.get("bearing") is not None:
                    flee_yaw = event["bearing"] + math.pi
                    if event.get("rear"):
                        self.rear_bearing = event["bearing"]
                        self.rear_bearing_age = obs["tick"]
                break
        if self.rear_bearing is not None and obs["tick"] - self.rear_bearing_age > 40:
            self.rear_bearing = None

        state = "zone"
        if not me["in_zone"]:
            state = "out_of_zone"
        elif target is not None:
            target_dist = math.hypot(target["pos"][0] - x, target["pos"][1] - y)
            if hp_frac < 0.32 or target_dist < 18:
                state = "disengage"
            else:
                state = "engage"
        elif obs["cubes"]:
            state = "collect"
        self.last_state = state

        if state == "out_of_zone":
            dx, dy = zc[0] - x, zc[1] - y
            dl = math.hypot(dx, dy) or 1.0
            act["move"] = [dx / dl, dy / dl]
            act["aim"] = [math.atan2(dy, dx), 0.0]
            act["boost"] = bool(me["energy"] > 12)
        elif state in ("engage", "disengage"):
            direction = self._best_movement_direction(me, target, tgt, nr, obs)
            act["move"] = direction or [0.0, 0.0]
            aim, fire, reload = self._combat_fire(me, target)
            if aim:
                act["aim"] = aim
            elif flee_yaw is not None:
                act["aim"] = [flee_yaw, 0.0]
            act["fire"] = fire
            if reload:
                act["reload"] = True
            energy = float(me["energy"])
            act["boost"] = bool(me["grounded"] and energy > 25
                                and state == "disengage")
            if state == "disengage" and target:
                target_pos = target["pos"]
                replan = (self.cover_goal is None
                          or obs["tick"] >= self.next_cover_scan
                          or math.hypot(self.cover_goal[0] - x,
                                        self.cover_goal[1] - y) < 8)
                if replan:
                    threat = {"pos": target_pos}
                    self.cover_goal = self._find_cover(x, y, z, threat, nxt, nr)
                    self.next_cover_scan = obs["tick"] + self.COVER_CHECK
                if self.cover_goal:
                    dx, dy = self.cover_goal[0] - x, self.cover_goal[1] - y
                    dl = math.hypot(dx, dy) or 1.0
                    act["move"] = [dx / dl, dy / dl]
            if me["grounded"] and energy > 35:
                act["jump"] = bool(obs["tick"] % 30 == 0 and target
                                   and math.hypot(target["pos"][0] - x,
                                                  target["pos"][1] - y) < 55)
        else:
            self.cover_goal = None
            dx, dy = tgt[0] - x, tgt[1] - y
            dl = math.hypot(dx, dy) or 1.0
            act["move"] = [dx / dl, dy / dl]
            act["aim"] = [math.atan2(dy, dx), 0.0]
            silent = self._silent_factor(me)
            act["move"][0] *= silent
            act["move"][1] *= silent
            act["boost"] = False
            if state == "collect":
                cb = min(obs["cubes"],
                         key=lambda c: (
                             math.dist(c["pos"], me["pos"])
                             + (0 if math.dist(c["pos"], me["pos"]) < self.CUBE_GRAB else 20)
                         ),
                         default=None)
                if cb:
                    vx, vy = cb["pos"][0] - x, cb["pos"][1] - y
                    d = math.hypot(vx, vy) or 1.0
                    act["move"] = [vx / d * silent, vy / d * silent]
                    act["aim"] = [math.atan2(vy, vx), 0.0]

        take = self._take_weapon(me, obs)
        if take:
            act.update(take)

        # Stand on a landed supply drop or actively pursue a nearby one.
        landed_drop = None
        for dr in obs.get("drops", []):
            dd = math.dist(dr["pos"][:2], (x, y))
            if dr.get("lands_in", 1) <= 0 and dr.get("holder") in (None, me["id"]):
                if dd < dr.get("radius", 8) * 0.8:
                    act["move"] = [0.0, 0.0]
                    act["boost"] = False
                    landed_drop = dr
                elif dd < 100:
                    landed_drop = dr
        if landed_drop and landed_drop.get("holder") in (None, me["id"]) \
                and math.dist(landed_drop["pos"][:2], (x, y)) >= landed_drop.get("radius", 8) * 0.8:
            vx, vy = landed_drop["pos"][0] - x, landed_drop["pos"][1] - y
            n = math.hypot(vx, vy) or 1.0
            act["move"] = [vx / n, vy / n]
            act["boost"] = bool(me["grounded"] and me["energy"] > 25)

        # out of zone overrides everything: sprint back in
        if not me["in_zone"]:
            dx, dy = zc[0] - x, zc[1] - y
            dl = math.hypot(dx, dy) or 1.0
            act["move"] = [dx / dl, dy / dl]
            act["boost"] = bool(me["energy"] > 12)
            act["aim"] = [math.atan2(dy, dx), 0.0]

        mf.draw.label(
            f"{me['hp']:.0f}hp {len(threats)}T {state}{' map-ok' if self.map else ''}"
        )
        if self.cover_goal:
            mf.draw.point([self.cover_goal[0], self.cover_goal[1], z], "#40c0ff", 6)
        if target:
            mf.draw.line([x, y, z + 7], target["pos"], "#ff5040")
        return act


mf.run(Bot)