import os
import sys
import tempfile
import numpy as np
import open3d as o3d
import floorplan_sota as fp

def pc(arr):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(arr, dtype=float))
    return p

def wall(xs, ys, h):
    z = np.arange(0, h, 0.05)
    return np.array([(x, y, zz) for x, y in zip(xs, ys) for zz in z])

def synth_room(w=5.0, d=4.0, h=2.5):
    gx, gy = np.meshgrid(np.arange(0, w, 0.05), np.arange(0, d, 0.05))
    floor = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    ceiling = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, h)]
    xa, ya = np.arange(0, w, 0.05), np.arange(0, d, 0.05)
    walls = np.vstack([wall(xa, np.zeros_like(xa), h), wall(xa, np.full_like(xa, d), h),
                       wall(np.zeros_like(ya), ya, h), wall(np.full_like(ya, w), ya, h)])
    dx = np.arange(2.0, 3.0, 0.02)
    door = wall(dx, np.zeros_like(dx), 2.0)
    wy = np.arange(1.0, 2.5, 0.02)
    window = wall(np.full_like(wy, w), wy, h)
    ang = np.linspace(0, 2 * np.pi, 48)
    col = np.array([(2.5 + 0.15 * np.cos(a), 2.0 + 0.15 * np.sin(a), z)
                    for z in np.arange(0, h, 0.08) for a in ang])
    return {"floor": pc(floor), "ceiling": pc(ceiling), "walls": pc(walls),
            "door": pc(door), "window": pc(window), "column": pc(col)}

def main():
    cfg = fp.load_cfg(fp.DEFAULT_CFG)
    seg, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    for n, p in synth_room().items():
        o3d.io.write_point_cloud(os.path.join(seg, n + ".ply"), p)
    info = fp.build(seg, out, cfg)
    files = ["floor_plan.dxf", "floor_plan.pdf", "floor_plan.png", "model.ifc", "floorplan.json"]
    missing = [f for f in files if not os.path.isfile(os.path.join(out, f))]
    ok = not missing and 16.0 < info["area_m2"] < 23.0 and info["n_openings"] >= 1
    print("method:", info["method"], "area_m2:", info["area_m2"],
          "openings:", info["n_openings"], "columns:", info["n_columns"])
    if missing:
        print("MISSING:", missing)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
