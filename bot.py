import math

import mechfight as mf

from actions import combat_fire, led_aim, take_weapon
from decisions import (
    choose_state,
    find_cover,
    movement_direction,
    pick_upgrade,
    safe_point,
    select_target,
    silent_factor,
    update_tracking,
)
from map import TerrainMap


class Bot:
    name = "ver-4-bot"

    PANIC_DIST = 100.0
    FIGHT_DIST = 60.0
    CUBE_GRAB = 25.0
    ZONE_MARGIN = 40.0
    COVER_CHECK = 40

    def __init__(self, hello: mf.Hello):
        config = hello["config"]
        self.eye = float(config["mech"].get("eye_height", 8.0))
        self.cfg = config
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
            self.map = TerrainMap(
                hello["map"]["dir"],
                hello["map"].get("files", []),
            )
            mf.log(f"ver-4-bot: map loaded, {len(self.map.prisms)} prisms")
        except Exception as error:
            mf.log(f"ver-4-bot: no map ({error!r}); fallback navigation mode")

    # Compatibility wrappers keep the existing bot API easy to test.
    def _safe_point(self, x, y, next_center, next_radius):
        return safe_point(x, y, next_center, next_radius, self.ZONE_MARGIN)

    def _find_cover(self, x, y, z, threat, next_center, next_radius):
        return find_cover(
            self.map,
            self.cfg,
            self.eye,
            x,
            y,
            z,
            threat,
            next_center,
            next_radius,
        )

    def _silent_factor(self, me):
        return silent_factor(me, self.cfg)

    def _led_aim(self, me, target, weapon=None):
        return led_aim(me, target, self.cfg, self.eye, weapon)

    def _pick_upgrade(self, obs):
        return pick_upgrade(obs, self.cfg)

    def _take_weapon(self, me, obs):
        return take_weapon(me, obs, self.cfg)

    def _combat_fire(self, me, target):
        return combat_fire(me, target, self.map, self.cfg, self.eye)

    def _update_tracking(self, obs):
        memory = float(self.cfg.get("vision", {}).get("memory_s", 5.0))
        self.tracked_mechs = update_tracking(
            self.tracked_mechs,
            obs,
            memory,
        )

    def _select_target(self, me, obs):
        return select_target(me, obs, self.tracked_mechs)

    def _best_movement_direction(self, me, target, next_target, next_radius, obs):
        return movement_direction(me, target, next_target, obs)

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        hp_fraction = me["hp"] / me["max_hp"]
        action: mf.Action = {"tick": obs["tick"]}

        pick = self._pick_upgrade(obs)
        if pick is not None:
            action["pick"] = pick

        if self.previous_hp is not None and me["hp"] < self.previous_hp:
            self.last_damage_tick = int(obs["tick"])
        for event in reversed(obs.get("events", [])):
            if event.get("type") == "hit":
                self.last_hit_tick = int(obs["tick"])
                break
        self.previous_hp = me["hp"]
        self._update_tracking(obs)

        zone = obs["zone"]
        zone_center = zone["center"]
        next_center, next_radius = zone["next_center"], zone["next_radius"]
        target_point = self._safe_point(x, y, next_center, next_radius)
        target = self._select_target(me, obs)

        threats = []
        for tracked in self.tracked_mechs.values():
            distance = math.hypot(
                tracked["pos"][0] - x,
                tracked["pos"][1] - y,
            )
            if distance < self.PANIC_DIST:
                threats.append((distance, tracked))

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

        state = choose_state(me, obs, target)
        self.last_state = state

        if state == "out_of_zone":
            dx, dy = zone_center[0] - x, zone_center[1] - y
            length = math.hypot(dx, dy) or 1.0
            action["move"] = [dx / length, dy / length]
            action["aim"] = [math.atan2(dy, dx), 0.0]
            action["boost"] = bool(me["energy"] > 12)
        elif state in ("engage", "disengage"):
            direction = self._best_movement_direction(
                me,
                target,
                target_point,
                next_radius,
                obs,
            )
            action["move"] = direction or [0.0, 0.0]
            aim, fire, reload = self._combat_fire(me, target)
            if aim:
                action["aim"] = aim
            elif flee_yaw is not None:
                action["aim"] = [flee_yaw, 0.0]
            action["fire"] = fire
            if reload:
                action["reload"] = True

            energy = float(me["energy"])
            action["boost"] = bool(
                me["grounded"]
                and energy > 25
                and state == "disengage"
            )
            if state == "disengage" and target:
                replan = (
                    self.cover_goal is None
                    or obs["tick"] >= self.next_cover_scan
                    or math.hypot(
                        self.cover_goal[0] - x,
                        self.cover_goal[1] - y,
                    ) < 8
                )
                if replan:
                    threat = {"pos": target["pos"]}
                    self.cover_goal = self._find_cover(
                        x,
                        y,
                        z,
                        threat,
                        next_center,
                        next_radius,
                    )
                    self.next_cover_scan = obs["tick"] + self.COVER_CHECK
                if self.cover_goal:
                    dx, dy = self.cover_goal[0] - x, self.cover_goal[1] - y
                    length = math.hypot(dx, dy) or 1.0
                    action["move"] = [dx / length, dy / length]

            if me["grounded"] and energy > 35:
                target_distance = (
                    math.hypot(target["pos"][0] - x, target["pos"][1] - y)
                    if target
                    else float("inf")
                )
                action["jump"] = bool(
                    obs["tick"] % 30 == 0
                    and target
                    and target_distance < 55
                )
        else:
            self.cover_goal = None
            dx, dy = target_point[0] - x, target_point[1] - y
            length = math.hypot(dx, dy) or 1.0
            action["move"] = [dx / length, dy / length]
            action["aim"] = [math.atan2(dy, dx), 0.0]
            silent = self._silent_factor(me)
            action["move"][0] *= silent
            action["move"][1] *= silent
            action["boost"] = False
            if state == "collect":
                cube = min(
                    obs["cubes"],
                    key=lambda item: (
                        math.dist(item["pos"], me["pos"])
                        + (0 if math.dist(item["pos"], me["pos"]) < self.CUBE_GRAB else 20)
                    ),
                    default=None,
                )
                if cube:
                    vx, vy = cube["pos"][0] - x, cube["pos"][1] - y
                    distance = math.hypot(vx, vy) or 1.0
                    action["move"] = [vx / distance * silent, vy / distance * silent]
                    action["aim"] = [math.atan2(vy, vx), 0.0]

        take = self._take_weapon(me, obs)
        if take:
            action.update(take)

        landed_drop = None
        for supply_drop in obs.get("drops", []):
            distance = math.dist(supply_drop["pos"][:2], (x, y))
            if (
                supply_drop.get("lands_in", 1) <= 0
                and supply_drop.get("holder") in (None, me["id"])
            ):
                radius = supply_drop.get("radius", 8) * 0.8
                if distance < radius:
                    action["move"] = [0.0, 0.0]
                    action["boost"] = False
                    landed_drop = supply_drop
                elif distance < 100:
                    landed_drop = supply_drop
        if (
            landed_drop
            and landed_drop.get("holder") in (None, me["id"])
            and math.dist(landed_drop["pos"][:2], (x, y))
            >= landed_drop.get("radius", 8) * 0.8
        ):
            vx, vy = landed_drop["pos"][0] - x, landed_drop["pos"][1] - y
            length = math.hypot(vx, vy) or 1.0
            action["move"] = [vx / length, vy / length]
            action["boost"] = bool(me["grounded"] and me["energy"] > 25)

        if not me["in_zone"]:
            dx, dy = zone_center[0] - x, zone_center[1] - y
            length = math.hypot(dx, dy) or 1.0
            action["move"] = [dx / length, dy / length]
            action["boost"] = bool(me["energy"] > 12)
            action["aim"] = [math.atan2(dy, dx), 0.0]

        mf.draw.label(
            f"{me['hp']:.0f}hp {len(threats)}T {state}"
            f"{' map-ok' if self.map else ''}"
        )
        if self.cover_goal:
            mf.draw.point(
                [self.cover_goal[0], self.cover_goal[1], z],
                "#40c0ff",
                6,
            )
        if target:
            mf.draw.line([x, y, z + 7], target["pos"], "#ff5040")
        return action


mf.run(Bot)
