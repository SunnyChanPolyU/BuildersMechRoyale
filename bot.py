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
            for base, top, bb, ring in self.prisms:
                if not (bb[0] <= px <= bb[2] and bb[1] <= py <= bb[3]):
                    continue
                if top <= pz or base >= pz:
                    continue
                if self._in_poly(px, py, ring):
                    return True
        return False


# -------------------------------------------------------------------- the bot

class Bot:
    name = "ver-2-bot"

    PANIC_DIST = 100.0   # flee if a mech is visible closer than this
    FIGHT_DIST = 60.0    # engage when pressed closer than this
    CUBE_GRAB = 25.0     # grab cubes within this range while calm
    ZONE_MARGIN = 40.0   # preferred inset from the next circle's edge
    COVER_CHECK = 40     # re-plan cover every N ticks (~2 s)

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.eye = float(cfg["mech"].get("eye_height", 8.0))
        self.map = None
        try:
            self.map = TerrainMap(hello["map"]["dir"],
                                  hello["map"].get("files", []))
            mf.log(f"ver-2-bot: map loaded, {len(self.map.prisms)} prisms")
        except Exception as e:  # any failure: degrade, never crash
            mf.log(f"ver-2-bot: no map ({e!r}); fallback flee mode")
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
                cz = self.map.ground(cx, cy) + self.eye
                if not self.map.blocked(tx, ty, tz + self.eye, cx, cy, cz):
                    continue
                cost = (math.hypot(cx - x, cy - y) * 0.6
                        + math.hypot(cx - tx, cy - ty) * 0.4)
                if cost < best_cost:
                    best, best_cost = (cx, cy), cost
        return best

    # ---- one decision -----------------------------------------------------

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        hp_frac = me["hp"] / me["max_hp"]
        act: mf.Action = {"tick": obs["tick"]}

        # upgrades: armor / sensors / mobility, else default 0
        if obs["offer"]:
            act["pick"] = 0
            for i, c in enumerate(obs["offer"]["choices"]):
                cid = c.get("id", "")
                if cid in ("armor", "sensors") or "speed" in cid or "boost" in cid:
                    act["pick"] = i
                    break

        zone = obs["zone"]
        zc = zone["center"]
        nxt, nr = zone["next_center"], zone["next_radius"]
        tgt = self._safe_point(x, y, nxt, nr)

        threats = sorted(
            ((math.dist(m["pos"], me["pos"]), m)
             for m in obs["mechs"] if m.get("kind") == "mech"),
            key=lambda t: t[0])
        near = [(d, m) for d, m in threats if d < self.PANIC_DIST]

        # unseen shooter: run opposite the last damage bearing
        flee_yaw = None
        for e in reversed(obs["events"]):
            if e.get("type") == "damaged" and e.get("weapon") != "zone":
                if e.get("bearing") is not None:
                    flee_yaw = e["bearing"] + math.pi
                break

        fleeing = bool(near) or flee_yaw is not None

        if fleeing:
            goal = self.cover_goal
            replan = (goal is None or obs["tick"] >= self.next_cover_scan
                      or math.hypot(goal[0] - x, goal[1] - y) < 8.0)
            if near and replan:
                self.cover_goal = self._find_cover(x, y, z, near[0][1], nxt, nr)
                self.next_cover_scan = obs["tick"] + self.COVER_CHECK
            goal = self.cover_goal or tgt

            # blended flee vector: toward goal, away from threat, away from hits
            vx, vy = 0.0, 0.0
            dg = math.hypot(goal[0] - x, goal[1] - y) or 1.0
            vx += (goal[0] - x) / dg
            vy += (goal[1] - y) / dg
            if near:
                d, m = near[0]
                t_yaw = math.atan2(m["pos"][1] - y, m["pos"][0] - x)
                w = 0.7 * (self.PANIC_DIST - min(d, self.PANIC_DIST)) / self.PANIC_DIST
                vx -= math.cos(t_yaw) * w
                vy -= math.sin(t_yaw) * w
            if flee_yaw is not None:
                vx += math.cos(flee_yaw) * 0.5
                vy += math.sin(flee_yaw) * 0.5
            n = math.hypot(vx, vy) or 1.0
            act["move"] = [vx / n, vy / n]
            act["boost"] = True

            if near:
                d, m = near[0]
                dxm, dym = m["pos"][0] - x, m["pos"][1] - y
                dh = math.hypot(dxm, dym) or 1.0
                apitch = math.atan2(m["pos"][2] + 4.5 - (z + 4.5), dh)
                act["aim"] = [math.atan2(dym, dxm), max(-0.3, min(0.3, apitch))]
                if d < self.FIGHT_DIST or hp_frac > 0.5:
                    act["fire"] = [True, True]
                act["jump"] = obs["tick"] % 30 == 0
            else:
                act["aim"] = [me["aim"][0], 0.0]
        else:
            self.cover_goal = None
            dx, dy = tgt[0] - x, tgt[1] - y
            dl = math.hypot(dx, dy) or 1.0
            act["move"] = [dx / dl, dy / dl]
            act["aim"] = [math.atan2(dy, dx), 0.0]
            act["boost"] = False
            if self.map is not None:
                act["move"][0] *= 0.7   # stay silent (below step_speed)
                act["move"][1] *= 0.7
                cb = min(obs["cubes"],
                         key=lambda c: math.dist(c["pos"], me["pos"]),
                         default=None)
                if cb and math.dist(cb["pos"], me["pos"]) < self.CUBE_GRAB:
                    vx, vy = cb["pos"][0] - x, cb["pos"][1] - y
                    n = math.hypot(vx, vy) or 1.0
                    act["move"] = [vx / n * 0.7, vy / n * 0.7]

        # stand on a landed supply drop we're capturing
        for dr in obs.get("drops", []):
            if dr.get("lands_in", 1) <= 0 and dr.get("holder") in (None, me["id"]) \
                    and math.dist(dr["pos"][:2], (x, y)) < dr.get("radius", 8) * 0.8:
                act["move"] = [0.0, 0.0]
                act["boost"] = False

        # out of zone overrides everything: sprint back in
        if not me["in_zone"]:
            dx, dy = zc[0] - x, zc[1] - y
            dl = math.hypot(dx, dy) or 1.0
            act["move"] = [dx / dl, dy / dl]
            act["boost"] = True
            act["aim"] = [math.atan2(dy, dx), 0.0]

        mode = "cover" if (fleeing and near) else ("flee" if fleeing else "hide")
        mf.draw.label(f"{me['hp']:.0f}hp {len(near)}T {mode}{' map-ok' if self.map else ''}")
        if self.cover_goal:
            mf.draw.point([self.cover_goal[0], self.cover_goal[1], z], "#40c0ff", 6)
        if near:
            mf.draw.line([x, y, z + 7], near[0][1]["pos"], "#ff5040")
        return act


mf.run(Bot)