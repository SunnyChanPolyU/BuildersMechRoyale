import math
import mechfight as mf


class Bot:
    name = "ver-1-bot"

    PANIC_DIST = 100.0
    FIGHT_DIST = 60.0

    def __init__(self, hello: mf.Hello):
        self.reach = hello["config"]["loot"]["pickup_radius"]
        self.margin = 40.0
        self.last_move = [1.0, 0.0]

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        hp_frac = me["hp"] / me["max_hp"]
        act: mf.Action = {"tick": obs["tick"]}

        # upgrades
        if obs["offer"]:
            best = 0
            for i, c in enumerate(obs["offer"]["choices"]):
                cid = c["id"]
                if cid in ("armor", "sensors") or "speed" in cid or "boost" in cid:
                    best = i
                    break
            act["pick"] = best

        zone = obs["zone"]
        zc, zr = zone["center"], zone["radius"]
        nxt, nr = zone["next_center"], zone["next_radius"]
        tgt = self._safe_point(x, y, nxt, nr)

        threats = sorted(
            ((math.dist(m["pos"], me["pos"]), m)
             for m in obs["mechs"] if m["kind"] == "mech"),
            key=lambda t: t[0],
        )
        near = [(d, m) for d, m in threats if d < self.PANIC_DIST]

        # unseen shooter: run away from the damage bearing
        flee_yaw = None
        for e in reversed(obs["events"]):
            if e.get("type") == "damaged" and e.get("weapon") != "zone":
                if e.get("bearing") is not None:
                    flee_yaw = e["bearing"] + math.pi
                break

        fleeing = bool(near) or flee_yaw is not None

        if fleeing:
            vx, vy = 0.0, 0.0
            if near:
                d, m = near[0]
                t_yaw = math.atan2(m["pos"][1] - y, m["pos"][0] - x)
                w = (self.PANIC_DIST - d) / self.PANIC_DIST
                vx -= math.cos(t_yaw) * w
                vy -= math.sin(t_yaw) * w
            if flee_yaw is not None:
                vx += math.cos(flee_yaw) * 0.6
                vy += math.sin(flee_yaw) * 0.6
            dx, dy = tgt[0] - x, tgt[1] - y
            dl = math.hypot(dx, dy) or 1.0
            vx += (dx / dl) * 0.5
            vy += (dy / dl) * 0.5
            n = math.hypot(vx, vy) or 1.0
            move = [vx / n, vy / n]
            self.last_move = move
            act["move"] = move
            act["boost"] = True
            if near:
                d, m = near[0]
                dxm, dym = m["pos"][0] - x, m["pos"][1] - y
                dh = math.hypot(dxm, dym) or 1.0
                ayaw = math.atan2(dym, dxm)
                apitch = math.atan2(m["pos"][2] + 4.5 - (z + 4.5), dh)
                act["aim"] = [ayaw, max(-0.3, min(0.3, apitch))]
                # engage only when pressed close or healthy enough for a potshot
                if d < self.FIGHT_DIST or hp_frac > 0.5:
                    act["fire"] = [True, True]
                act["jump"] = obs["tick"] % 30 == 0
            else:
                act["aim"] = [me["aim"][0], 0.0]
        else:
            dx, dy = tgt[0] - x, tgt[1] - y
            dl = math.hypot(dx, dy) or 1.0
            act["move"] = [dx / dl, dy / dl]
            act["aim"] = [math.atan2(dy, dx), 0.0]
            act["boost"] = False
            # quietly grab a nearby cube (no detour into danger)
            cb = min(obs["cubes"], key=lambda c: math.dist(c["pos"], me["pos"]),
                     default=None)
            if cb and math.dist(cb["pos"], me["pos"]) < 25.0:
                cx, cy = cb["pos"][:2]
                vx, vy = cx - x, cy - y
                n = math.hypot(vx, vy) or 1.0
                act["move"] = [vx / n, vy / n]

        # capture a landed supply drop if we're alone on it
        for dr in obs["drops"]:
            if dr.get("lands_in", 1) <= 0 and dr.get("holder") in (None, me["id"]) \
                    and math.dist(dr["pos"][:2], (x, y)) < dr.get("radius", 8) * 0.8:
                act["move"] = [0.0, 0.0]
                act["boost"] = False

        # sprint back into the zone
        if not me["in_zone"]:
            dx, dy = zc[0] - x, zc[1] - y
            dl = math.hypot(dx, dy) or 1.0
            act["move"] = [dx / dl, dy / dl]
            act["boost"] = True
            act["aim"] = [math.atan2(dy, dx), 0.0]

        mode = "flee" if fleeing else "hide"
        mf.draw.label(f"{me['hp']:.0f}hp · {len(near)} threats · {mode}")
        mf.draw.point([tgt[0], tgt[1], z], "#40c0ff", 6)
        return act

    def _safe_point(self, x, y, nxt, nr):
        """Comfortable interior point of the next circle."""
        dx, dy = nxt[0] - x, nxt[1] - y
        d = math.hypot(dx, dy)
        r_target = max(nr - self.margin, 0.0)
        if d <= r_target:
            return tuple(nxt)
        return (nxt[0], nxt[1])


mf.run(Bot)