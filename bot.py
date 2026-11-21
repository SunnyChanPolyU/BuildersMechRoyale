import json
import math
import os
import struct

import mechfight as mf


# ------------------------------------------------------------------ map model

class TerrainMap:
    """Ground heights + sight-blocking prisms. Loaded from hello.map.dir."""

    MAX_SEG_SAMPLES = 40   # perf cap: a 280 m ray at 3 m steps is ~93 -> clamp

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
            if p.get("kind") == "bridge":
                continue
            ring = p["ring"]
            xs, ys = ring[0::2], ring[1::2]
            self.prisms.append((p["base"], p["top"],
                                (min(xs), min(ys), max(xs), max(ys)),
                                list(zip(xs, ys))))

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

    def blocked(self, x1, y1, z1, x2, y2, z2, step=3.0):
        """Any prism wall crossing the segment (x1,y1,z1)->(x2,y2,z2)?

        step 3.0 (was 4.0) plus a sample cap so a blocked() call can never
        blow the tick budget; thin-wall tunnelling is correspondingly less
        likely than before."""
        d = math.hypot(x2 - x1, y2 - y1)
        if d < 1e-6:
            return False
        n = max(2, min(int(d / step) + 1, self.MAX_SEG_SAMPLES))
        pris = self.prisms
        for k in range(1, n):
            u = k / n
            px = x1 + (x2 - x1) * u
            py = y1 + (y2 - y1) * u
            pz = z1 + (z2 - z1) * u
            for base, top, bb, ring in pris:
                if not (bb[0] <= px <= bb[2] and bb[1] <= py <= bb[3]):
                    continue
                if top <= pz or base >= pz:
                    continue
                if self._in_poly(px, py, ring):
                    return True
        return False


# -------------------------------------------------------------------- the bot

class Bot:
    name = "ver-3-bot"

    ZONE_MARGIN = 40.0   # preferred inset from the next circle's edge
    COVER_CHECK = 40     # re-plan cover every N ticks (~2 s)
    ENERGY_BOOST_MIN = 25.0  # don't boost below this energy
    JUMP_ENERGY = 30.0      # keep this much in reserve for a jump
    CUBE_DETOUR = 80.0      # calm: detour to cubes within this range
    LOOT_DETOUR = 150.0     # calm: detour to pickups/crates within this
    CRATE_RANGE = 140.0     # shoot crates within this (gun range is 230)
    AIM_TOL = 0.15          # rad: torso must be this close before firing
    MAX_THREAT_MEM = 5.0    # seconds to remember sound/beam threats

    # weapon value for take/keep/swap decisions
    W_RANK = {"laser": 4, "lightning": 3, "mortar": 2, "gun": 1}

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.cfg = cfg
        mch = cfg.get("mech", {})
        snd = cfg.get("sound", {})

        # Live config instead of assumptions (balance changes between matches)
        self.step_speed = float(snd.get("step_speed", 5.0))
        def_w = cfg.get("weapons", {})
        # longest enemy engagement range we can face
        rngs = []
        for k in ("laser", "lightning", "mortar", "gun"):
            wd = def_w.get(k)
            if isinstance(wd, dict):
                rngs.append(float(wd.get("range", 230)))
        self.max_weapon_range = max(rngs) if rngs else 280.0
        self.gun_speed = float(def_w.get("gun", {}).get("speed", 252))
        self.min_pitch = float(mch.get("min_pitch", -0.35))
        self.max_pitch = float(mch.get("max_pitch", 0.35))

        # fight/panic radii derived from weapon data, not magic numbers:
        # panic out to ~1/3 of the longest range but at least 90 m,
        # fight at half the panic distance.
        self.panic_dist = max(90.0, self.max_weapon_range / 3.0)
        self.fight_dist = self.panic_dist / 2.0

        self.pickup_radius = float(cfg.get("loot", {}).get("pickup_radius", 4.0))
        self.energy_jet_floor = float(mch.get("jump_speed", 16.0))

        # upgrade metadata: id -> scored string blob of its effects
        self.upgrade_score = {}
        try:
            for u in cfg.get("upgrades", []):
                uid, blob = u.get("id"), json.dumps(u)
                score = 0
                low = blob.lower()
                if "damage" in low:
                    score += 4            # weapon damage / raw output
                if "cooldown" in low and '"mul"' in low:
                    score += 3            # faster fire
                if "taken" in low or "armor" in uid:
                    score += 4            # survivability
                if "heal" in low or "regen" in low:
                    score += 3
                if "cube" in low:
                    score += 2            # economy -> more offers
                if "speed" in low:
                    score += 2
                if "mag" in low or "reload" in low:
                    score += 1
                if "sensor" in uid or "vision" in low or "sound" in low:
                    score += 1
                self.upgrade_score[uid] = score
        except Exception:
            pass

        self.map = None
        try:
            self.map = TerrainMap(hello["map"]["dir"],
                                 hello["map"].get("files", []))
            mf.log(f"ver-3-bot: map loaded, {len(self.map.prisms)} prisms")
        except Exception as e:  # any failure: degrade, never crash
            mf.log(f"ver-3-bot: no map ({e!r}); fallback flee mode")
        self.cover_goal = None
        self.next_cover_scan = 0
        # remembered unseen-threat bearings: (tick_when, world yaw, weight)
        self.threat_mem = []

    # ---- helpers ----------------------------------------------------------

    def _eye_h(self, me):
        """Live eye height (cubes scale you up; don't use the base config)."""
        st = me.get("stats", {})
        return float(st.get("eye_height", 8.0))

    def _silent_len(self, me):
        """Move-vector length that keeps ground speed below step_speed,
        computed from live values, with a safety margin."""
        walk = float(me.get("stats", {}).get("walk_speed", 11.7)) or 11.7
        return max(0.1, min(1.0, 0.85 * self.step_speed / walk))

    def _weapon_slots(self, me):
        """[(slot_idx, kind)] for non-null slots."""
        out = []
        for i, w in enumerate(me.get("weapons", [])):
            if w:
                k = w.get("kind") or (w.get("stats", {}) or {}).get("kind", "")
                out.append((i, k))
        return out

    def _slot_stats(self, me, slot):
        w = me.get("weapons", [None, None])[slot]
        return (w or {}).get("stats", {}) or {}

    # ---- goal points ------------------------------------------------------

    def _safe_point(self, x, y, nxt, nr):
        """A point on the inner rim of the next circle, biased toward our
        side of it: keeps us off the bare centre (open, contested ground)
        while guaranteeing we are ZONE_MARGIN inside the shrinking edge."""
        r = max(nr - self.ZONE_MARGIN, 0.0)
        dx, dy = nxt[0] - x, nxt[1] - y
        d = math.hypot(dx, dy) or 1.0
        if d > r:
            return (nxt[0] - (dx / d) * r, nxt[1] - (dy / d) * r)
        # already inside the target disc: hold our current bearing from the
        # centre, pulled out to the rim (camp rim+cover, not the centre)
        ox, oy = -dx, -dy
        od = math.hypot(ox, oy) or 1.0
        if od < 5.0:            # basically at the centre: pick any rim spot
            ox, oy, od = 1.0, 0.0, 1.0
        return (nxt[0] + ox / od * r, nxt[1] + oy / od * r)

    def _find_cover(self, x, y, z, threat, nxt, nr, eye):
        """Spot where a prism blocks LOS from `threat`, inside the next
        circle. Fewer candidates than before and never re-run every tick
        when it fails: the caller caches a fallback retreat point instead."""
        if self.map is None:
            return None
        tx, ty, tz = threat["pos"]
        b = self.map.bounds
        teye = tz + 8.0       # assume standard enemy eye height
        best, best_cost = None, float("inf")
        for i in range(16):                       # 16 x 2 = 32 candidates
            ang = i * (2 * math.pi / 16) + 0.196
            for r in (25.0, 55.0):
                cx = x + r * math.cos(ang)
                cy = y + r * math.sin(ang)
                if not (b["minX"] < cx < b["maxX"] and b["minY"] < cy < b["maxY"]):
                    continue
                if math.hypot(cx - nxt[0], cy - nxt[1]) > nr - 10.0:
                    continue          # don't hide into the closing storm
                cz = self.map.ground(cx, cy) + eye
                if not self.map.blocked(tx, ty, teye, cx, cy, cz):
                    continue
                cost = (math.hypot(cx - x, cy - y) * 0.6
                        + math.hypot(cx - tx, cy - ty) * 0.4)
                if cost < best_cost:
                    best, best_cost = (cx, cy), cost
        return best

    # ---- senses ------------------------------------------------------------

    def _remember_threats(self, obs):
        """Feed sounds + beams + damage events into short-term threat
        memory. Sounds/beams reveal shooters we can't see (free intel);
        beams pin the shooter at path[0] (the muzzle)."""
        t = obs["t"]
        mem = self.threat_mem
        for e in obs.get("events", []):
            if e.get("type") == "damaged" and e.get("bearing") is not None:
                mem.append((t, e["bearing"], 1.0))
        for s in obs.get("sounds", []):
            if s.get("kind") in ("gun", "laser", "laser_charge",
                                 "lightning", "npc_gun") and s.get("bearing") is not None:
                w = 0.7 if s.get("dist") == "near" else 0.3
                mem.append((t, s["bearing"], w))
        for bm in obs.get("beams", []):
            p = bm.get("path") or []
            if p:
                mem.append((t, math.atan2(p[0][1] - obs["self"]["pos"][1],
                                          p[0][0] - obs["self"]["pos"][0]), 1.0))
        # expire
        self.threat_mem = [m for m in mem if t - m[0] < self.MAX_THREAT_MEM][-8:]

    def _lead_aim(self, me, tgt, shot_speed):
        """Aim at where the target will be when the shot arrives, and at
        its body centre. Returns (yaw, pitch) or None if hopeless."""
        mx, my, mz = me["pos"]
        tx, ty, tz = tgt["pos"]
        dx, dy = tx - mx, ty - my
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return None
        vx = (tgt.get("vel") or (0, 0))[0]
        vy = (tgt.get("vel") or (0, 0))[1]
        flight = d / max(shot_speed, 1.0)
        lx, ly = tx + vx * flight, ty + vy * flight
        # reject impossible leads: aim point farther than a body radius off
        # a 4 m/s target still lands inside the cylinder; huge boosts may miss
        return math.atan2(ly - my, lx - mx), tz + 4.5

    # ---- one decision -----------------------------------------------------

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs["t"]
        hp_frac = me["hp"] / me["max_hp"]
        energy = float(me.get("energy", 100))
        eye = self._eye_h(me)
        act: mf.Action = {"tick": obs["tick"]}

        # ---------------- upgrades: score by real effects, not id strings
        if obs["offer"]:
            best_i, best_s = 0, -1
            for i, c in enumerate(obs["offer"]["choices"]):
                s = self.upgrade_score.get(c.get("id", ""), 0)
                if s > best_s:
                    best_i, best_s = i, s
            act["pick"] = best_i

        self._remember_threats(obs)
        threat_yaws = [m[1] for m in self.threat_mem]

        zone = obs["zone"]
        zc = zone["center"]
        nxt, nr = zone["next_center"], zone["next_radius"]
        tgt = self._safe_point(x, y, nxt, nr)

        # visible threats: mechs everywhere in panic radius; also NPC drones
        # close in (they shoot too, 350 HP is cheap XP and they hurt).
        threats = []
        for m in obs["mechs"]:
            d = math.dist(m["pos"], me["pos"])
            if m.get("kind") == "mech" and d < self.panic_dist:
                threats.append((d, m))
            elif m.get("kind") == "npc" and d < self.fight_dist:
                threats.append((d, m))
        threats.sort(key=lambda tm: tm[0])
        near = threats

        fleeing = bool(near) or bool(threat_yaws)
        # recent threat memory only forces cover-seeking, not full panic
        if not near and threat_yaws and t - self.threat_mem[-1][0] > 2.0:
            fleeing = False if hp_frac > 0.5 else fleeing

        # ---------------- fleeing / fighting ------------------------------
        if fleeing:
            goal = self.cover_goal
            replan = (goal is None or obs["tick"] >= self.next_cover_scan
                      or math.hypot(goal[0] - x, goal[1] - y) < 8.0)
            if near and replan:
                self.cover_goal = self._find_cover(
                    x, y, z, near[0][1], nxt, nr, eye)
                self.next_cover_scan = obs["tick"] + self.COVER_CHECK
                if self.cover_goal is None:
                    # fallback: straight away from the threat, 45 m out,
                    # kept inside the circle. Cached so a failed scan does
                    # NOT re-trigger every tick
                    th = near[0][1]["pos"]
                    a = math.atan2(y - th[1], x - th[0])
                    fx, fy = x + 45 * math.cos(a), y + 45 * math.sin(a)
                    dc = math.hypot(fx - nxt[0], fy - nxt[1])
                    if dc > nr - 15:
                        s = max((nr - 15) / dc, 0.0)
                        fx = nxt[0] + (fx - nxt[0]) * s
                        fy = nxt[1] + (fy - nxt[1]) * s
                    self.cover_goal = (fx, fy)
            goal = self.cover_goal or tgt

            # blended flee vector: toward cover, away from threat, away
            # from remembered shooter bearings.
            vx, vy = 0.0, 0.0
            dg = math.hypot(goal[0] - x, goal[1] - y) or 1.0
            vx += (goal[0] - x) / dg
            vy += (goal[1] - y) / dg
            if near:
                d, m = near[0]
                t_yaw = math.atan2(m["pos"][1] - y, m["pos"][0] - x)
                w = 0.7 * (self.panic_dist - min(d, self.panic_dist)) / self.panic_dist
                vx -= math.cos(t_yaw) * w
                vy -= math.sin(t_yaw) * w
            for ty_ in threat_yaws:
                vx += math.cos(ty_ + math.pi) * 0.25
                vy += math.sin(ty_ + math.pi) * 0.25
            n = math.hypot(vx, vy) or 1.0
            act["move"] = [vx / n, vy / n]
            act["boost"] = energy > self.ENERGY_BOOST_MIN
            act["jump"] = False
            # dodge incoming mortar shells (concussion: knock + 45 energy)
            for p in obs.get("projectiles", []):
                if p.get("kind") != "mortar":
                    continue
                px, py = p["pos"][0], p["pos"][1]
                vx_, vy_ = p.get("vel") or (0, 0)
                rx, ry = x - px, y - py
                rd = math.hypot(rx, ry)
                dot = (rx * vx_ + ry * vy_) / (rd * (math.hypot(vx_, vy_) or 1.0))
                if rd < 60 and dot > 0.85:      # shell roughly heading at us
                    pvx, pvy = -ry, rx          # perpendicular to its velocity
                    pn = math.hypot(pvx, pvy) or 1.0
                    act["move"] = [(act["move"][0] + pvx / pn),
                                   (act["move"][1] + pvy / pn)]
                    mn = math.hypot(*act["move"]) or 1.0
                    act["move"] = [act["move"][0] / mn, act["move"][1] / mn]
                    break

            if near:
                d, m = near[0]
                aim = self._lead_aim(me, m, self.gun_speed)
                if aim:
                    yaw, az = aim
                    dh = math.hypot(m["pos"][0] - x, m["pos"][1] - y)
                    pitch = math.atan2(az - (z + 4.5), dh)
                    act["aim"] = [yaw, max(self.min_pitch, min(self.max_pitch, pitch))]
                    # fire only with LOS, in range, and once the torso has
                    # actually slewed onto the target; no long-range potshots
                    clear = (self.map is None
                             or not self.map.blocked(x, y, z + eye,
                                                     m["pos"][0], m["pos"][1],
                                                     m["pos"][2] + 4.5))
                    dyaw = abs(((yaw - me["aim"][0] + math.pi)
                                % (2 * math.pi)) - math.pi)
                    if (clear and d < self.fight_dist
                            and dyaw < self.AIM_TOL and energy > 5):
                        act["fire"] = self._fire_pattern(me)
                    else:
                        act["fire"] = [False, False]
                else:
                    act["aim"] = [me["aim"][0], 0.0]
                    act["fire"] = [False, False]
            else:
                # unseen shooter: turn TOWARD the freshest bearing so we
                yaw = threat_yaws[-1] if threat_yaws else me["aim"][0]
                act["aim"] = [yaw, 0.0]
                act["fire"] = [False, False]

        # ---------------- calm behaviour: loot, cubes, crates --------------
        else:
            self.cover_goal = None
            sl = self._silent_len(me)          # truly below step_speed
            move_goal = tgt
            move_len = sl

            # detour priorities: weapon pickup > cube > crate loot > zone
            best_pick = None
            my_rank = max((self.W_RANK.get(k, 0) for _, k in self._weapon_slots(me)),
                          default=0)
            for pk in obs.get("pickups", []):
                d = math.dist(pk["pos"], me["pos"])
                if d > self.LOOT_DETOUR:
                    continue
                rank = self.W_RANK.get(pk.get("kind", ""), 0)
                if rank == 0:
                    continue
                # worth it if it's an upgrade for us (or our only gun got dropped)
                worth = rank > my_rank or not self._weapon_slots(me)
                if worth and (best_pick is None or d < best_pick[0]):
                    best_pick = (d, pk)

            cb = min(obs["cubes"],
                     key=lambda c: math.dist(c["pos"], me["pos"]),
                     default=None)
            crate = min(obs["crates"],
                        key=lambda c: math.dist(c["pos"], me["pos"]),
                        default=None)

            if best_pick and best_pick[0] < self.LOOT_DETOUR:
                move_goal = (best_pick[1]["pos"][0], best_pick[1]["pos"][1])
                if best_pick[0] <= self.pickup_radius + 1.5:
                    # in range: take it (swap out our worst slot if needed)
                    act["take"] = best_pick[1]["id"]
                    slots = self._weapon_slots(me)
                    if len(slots) >= 2 and not any(s is None
                                                   for s in me.get("weapons", [])):
                        worst = min(slots, key=lambda sk: self.W_RANK.get(sk[1], 0))
                        act["drop"] = worst[0]
            elif cb and math.dist(cb["pos"], me["pos"]) < self.CUBE_DETOUR:
                move_goal = (cb["pos"][0], cb["pos"][1])
            elif crate and math.dist(crate["pos"], me["pos"]) < self.CRATE_RANGE:
                # crates give cubes + half carry a weapon: shoot them open
                move_goal = (crate["pos"][0], crate["pos"][1])
                cpos = crate["pos"]
                dh = math.hypot(cpos[0] - x, cpos[1] - y)
                if dh > self.pickup_radius + 6:
                    clear = (self.map is None
                             or not self.map.blocked(x, y, z + eye,
                                                     cpos[0], cpos[1],
                                                     cpos[2] + 2.0))
                    yaw = math.atan2(cpos[1] - y, cpos[0] - x)
                    act["aim"] = [yaw, 0.0]
                    dyaw = abs(((yaw - me["aim"][0] + math.pi)
                                % (2 * math.pi)) - math.pi)
                    if clear and dyaw < self.AIM_TOL and dh < 200:
                        act["fire"] = self._fire_pattern(me)
                        act["move"] = [0.0, 0.0]   # stand still to shoot
                else:
                    act["aim"] = [me["aim"][0], 0.0]

            if "move" not in act:
                dx, dy = move_goal[0] - x, move_goal[1] - y
                dl = math.hypot(dx, dy) or 1.0
                act["move"] = [dx / dl * move_len, dy / dl * move_len]
            if "aim" not in act:
                act["aim"] = [math.atan2(move_goal[1] - y, move_goal[0] - x), 0.0]
            act["boost"] = False
            act["fire"] = act.get("fire", [False, False])

            # proactive gun reload while nobody is looking
            g = me["weapons"][0] if me.get("weapons") else None
            if g and g.get("kind") == "gun":
                if g.get("ammo", 12) <= 2 and not g.get("reload"):
                    act["reload"] = True

        if not fleeing:
            for dr in obs.get("drops", []):
                if dr.get("lands_in", 1) <= 0 \
                        and dr.get("holder") in (None, me["id"]) \
                        and math.dist(dr["pos"][:2], (x, y)) < dr.get("radius", 8) * 0.8:
                    # hold the ring, but keep aiming/firing: standing frozen
                    # under fire was a 4-second free kill.
                    act["move"] = [0.0, 0.0]
                    act["boost"] = False
                    break

        # ---------------- out of zone overrides everything -----------------
        if not me["in_zone"]:
            dx, dy = zc[0] - x, zc[1] - y
            dl = math.hypot(dx, dy) or 1.0
            act["move"] = [dx / dl, dy / dl]
            act["boost"] = energy > 10     # spend the tank only to survive
            act["aim"] = [math.atan2(dy, dx), 0.0]

        mode = "fight" if fleeing else "loot"
        mf.draw.label(f"{me['hp']:.0f}hp e{energy:.0f} "
                      f"{len(near)}T {mode}{' map-ok' if self.map else ''}")
        if self.cover_goal:
            mf.draw.point([self.cover_goal[0], self.cover_goal[1], z], "#40c0ff", 6)
        if near:
            mf.draw.line([x, y, z + 7], near[0][1]["pos"], "#ff5040")
        return act

    def _fire_pattern(self, me):
        """fire=[slot0, slot1] honouring laser charge semantics."""
        fire = [False, False]
        ws = me.get("weapons", [])
        for i, w in enumerate(ws):
            if not w or i > 1:
                continue
            k = w.get("kind", "")
            if k == "laser":
                st = self._slot_stats(me, i)
                full = float(st.get("charge", 0.75))
                ch = float(w.get("charge", 0.0))
                fire[i] = ch < full * 0.9   # hold to charge, release fires
            else:
                fire[i] = float(w.get("cooldown", 0.0)) <= 0.02
        return fire


mf.run(Bot)