import json
import math
import os
import struct


class TerrainMap:
    """Ground heights and sight-blocking map prisms."""

    def __init__(self, map_dir: str, map_files):
        paths = {
            os.path.basename(path): os.path.join(map_dir, path)
            for path in map_files
        }
        world_path = paths.get("world.json")
        terrain_path = paths.get("terrain.i16")
        if world_path is None or terrain_path is None:
            raise OSError(f"map manifest lacks expected files: {map_files}")

        with open(world_path, encoding="utf-8") as handle:
            world = json.load(handle)

        terrain = world["terrain"]
        self.tx, self.ty = terrain["minX"], terrain["minY"]
        self.tstep = terrain["step"]
        self.tnx, self.tny = terrain["nx"], terrain["ny"]
        self.scale = terrain.get("scale", 0.01)

        with open(terrain_path, "rb") as handle:
            raw = handle.read()
        required = 2 * self.tnx * self.tny
        if len(raw) < required:
            raise OSError(f"terrain file short: {len(raw)} < {required}")
        self.heights = struct.unpack(
            "<%dh" % (self.tnx * self.tny),
            raw[:required],
        )
        self.h = self.heights

        self.bounds = world["bounds"]
        self.prisms = []
        for prism in world.get("prisms", []):
            if prism.get("kind") == "bridge":
                continue

            ring = prism["ring"]
            xs, ys = ring[0::2], ring[1::2]
            holes_flat = prism.get("holes", [])
            holes = []
            if holes_flat and isinstance(holes_flat[0], (int, float)):
                holes_flat = [holes_flat]
            for hole in holes_flat:
                hx, hy = hole[0::2], hole[1::2]
                holes.append(list(zip(hx, hy)))

            self.prisms.append((
                prism["base"],
                prism["top"],
                (min(xs), min(ys), max(xs), max(ys)),
                list(zip(xs, ys)),
                holes,
            ))

    def ground(self, x: float, y: float) -> float:
        fx = (x - self.tx) / self.tstep
        fy = (y - self.ty) / self.tstep
        i0 = max(0, min(self.tnx - 2, int(fx)))
        j0 = max(0, min(self.tny - 2, int(fy)))
        ax = min(max(fx - i0, 0.0), 1.0)
        ay = min(max(fy - j0, 0.0), 1.0)

        def height(i, j):
            return self.heights[j * self.tnx + i] * self.scale

        return (
            (height(i0, j0) * (1 - ax) + height(i0 + 1, j0) * ax) * (1 - ay)
            + (height(i0, j0 + 1) * (1 - ax) + height(i0 + 1, j0 + 1) * ax) * ay
        )

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
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance < 1e-6:
            return False
        samples = max(2, int(distance / step))
        for k in range(1, samples):
            u = k / samples
            px = x1 + (x2 - x1) * u
            py = y1 + (y2 - y1) * u
            pz = z1 + (z2 - z1) * u
            for base, top, bounds, ring, holes in self.prisms:
                if not (bounds[0] <= px <= bounds[2] and bounds[1] <= py <= bounds[3]):
                    continue
                if top <= pz or base >= pz:
                    continue
                if self._in_poly(px, py, ring) and not any(
                    self._in_poly(px, py, hole) for hole in holes
                ):
                    return True
        return False

    def solid_at(self, x, y, feet_z, height=9.0):
        for base, top, bounds, ring, holes in self.prisms:
            if not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
                continue
            if not self._in_poly(x, y, ring):
                continue
            if any(self._in_poly(x, y, hole) for hole in holes):
                continue
            if base < feet_z + height and top > feet_z:
                return True
        return False

    def reachable(self, x, y, from_z, max_step=1.2, max_slope_deg=50.0):
        bounds = self.bounds
        if not (bounds["minX"] < x < bounds["maxX"] and bounds["minY"] < y < bounds["maxY"]):
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
