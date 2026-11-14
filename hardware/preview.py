import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from trimesh.intersections import mesh_plane

OUT = Path(__file__).resolve().parent / "out"
W = 520


def rot(yaw, pitch):
    a, b = np.radians(yaw), np.radians(pitch)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return rx @ rz


def shade(mesh, yaw, pitch, title, extra=None):
    R = rot(yaw, pitch)
    v = mesh.vertices @ R.T
    p = v[:, [0, 2]] * np.array([1, -1])
    lo, hi = p.min(0), p.max(0)
    k = (W - 60) / max(hi - lo)
    q = (p - lo) * k + 30
    img = np.full((W, W, 3), 250, np.uint8)
    tri = mesh.faces
    n = mesh.face_normals @ R.T
    depth = v[tri][:, :, 1].mean(1)
    light = np.array([0.4, -0.8, 0.45])
    light /= np.linalg.norm(light)
    for i in np.argsort(-depth):
        if n[i, 1] > 0:
            continue
        lum = 0.3 + 0.7 * max(0.0, float(np.dot(n[i], light)))
        c = np.array([70, 150, 230]) * lum
        cv2.fillConvexPoly(img, np.round(q[tri[i]] * 4).astype(np.int32), tuple(int(x) for x in c), cv2.LINE_AA, 2)
    cv2.putText(img, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def section(mesh, title):
    segs = mesh_plane(mesh, [1, 0, 0], [0, 0, 0])
    yz = segs[:, :, 1:]
    lo, hi = yz.reshape(-1, 2).min(0), yz.reshape(-1, 2).max(0)
    k = (W - 80) / max(hi - lo)
    img = np.full((W, W, 3), 250, np.uint8)
    for s in yz:
        a = ((s[0] - lo) * k + 40)
        b = ((s[1] - lo) * k + 40)
        cv2.line(img, (int(a[0]), int(W - a[1])), (int(b[0]), int(W - b[1])), (30, 30, 30), 1, cv2.LINE_AA)
    for mm in range(0, int(hi[0] - lo[0]) + 1, 10):
        x = int(mm * k + 40)
        cv2.line(img, (x, W - 20), (x, W - 14), (120, 120, 120), 1)
        cv2.putText(img, str(mm), (x - 6, W - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(img, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def assembly(falu, pencil_d=8.0, seat=1.35, spring_depth=9.25, spring_turn=105.0):
    import pencil_holder as ph
    ox, oy = ph.RING_AXIS
    mount = ph.from_stl(falu / ph.MOUNT_FILE)
    guide = ph.from_stl(OUT / "pencil_guide.stl").translate([ox, oy, seat])
    name = next(k for k, (_, d) in ph.SPRINGS.items() if pencil_d + 0.3 <= d)
    spring = ph.from_stl(falu / ph.SPRINGS[name][0]).translate([-24.2, 0, -10]).rotate([90, 0, 0]).rotate([0, 0, spring_turn])
    spring = spring.translate([ox, oy, seat + ph.LENGTH - spring_depth])
    r = pencil_d / 2
    pencil = ph.Manifold.cylinder(150, r, r, 48).translate([ox, oy, seat - 25])
    pencil = pencil + ph.Manifold.cylinder(6, 0.4, r, 48).translate([ox, oy, seat - 31])
    clashes = {
        "guide_in_mount_mm3": round((guide ^ mount).volume(), 2),
        "spring_on_guide_mm3": round((spring ^ guide).volume(), 2),
        "spring_against_mount_mm3": round((spring ^ mount).volume(), 2),
        "pencil_in_guide_mm3": round((pencil ^ guide).volume(), 2),
        "pencil_in_spring_mm3": round((pencil ^ spring).volume(), 2),
    }
    combo = trimesh.util.concatenate([ph.to_mesh(m) for m in (mount, guide, spring, pencil)])
    views = ((30, 15, "assembled - front left"), (120, 15, "assembled - clip side"), (0, 89.9, "assembled - from above"))
    image = np.hstack([shade(combo, yaw, pitch, title) for yaw, pitch, title in views])
    cv2.imwrite(str(OUT / "pencil_holder_assembly.png"), image)
    return {"spring_module": name, **clashes}


def main():
    guide = trimesh.load(OUT / "pencil_guide.stl")
    rings = trimesh.load(OUT / "pencil_fit_rings.stl")
    tiles = [shade(guide, 35, 20, "pencil_guide"), section(guide, "pencil_guide cut along its axis (mm)"),
             shade(rings, 35, 30, "pencil_fit_rings")]
    cv2.imwrite(str(OUT / "pencil_holder_preview.png"), np.hstack(tiles))
    falu = Path(__file__).resolve().parent / "falu"
    if falu.exists():
        print(assembly(falu))


if __name__ == "__main__":
    main()
