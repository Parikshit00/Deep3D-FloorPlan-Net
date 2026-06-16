import os
import json
import math
import datetime
import numpy as np
import yaml
import cv2
import open3d as o3d
from shapely.geometry import Polygon, Point
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.addons.drawing import matplotlib as dxf_mpl

CLASSES = ["floor", "ceiling", "walls", "window", "door", "column"]
DEFAULT_CFG = "../configs/floorplan.yml"
S = 1000.0

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

def load_structural(seg_dir):
    out = {}
    for n in CLASSES:
        p = os.path.join(seg_dir, n + ".ply")
        out[n] = o3d.io.read_point_cloud(p) if os.path.isfile(p) else o3d.geometry.PointCloud()
    return out

def xy(pcd):
    return np.asarray(pcd.points)[:, :2] if len(pcd.points) else np.empty((0, 2))

def vertical_wall_lines(boundary_pcd, w):
    pcd = boundary_pcd.voxel_down_sample(w["voxel"])
    if len(pcd.points) < w["min_inliers"]:
        return []
    lines, work = [], o3d.geometry.PointCloud(pcd)
    for _ in range(w["max_planes"]):
        if len(work.points) < w["min_inliers"]:
            break
        (a, b, c, d), inl = work.segment_plane(w["ransac_dist"], 3, 1000)
        if len(inl) < w["min_inliers"]:
            break
        ip = np.asarray(work.select_by_index(inl).points)
        if abs(c) < w["vertical_tol"]:
            nrm = math.hypot(a, b)
            nx, ny = a / nrm, b / nrm
            cx, cy = ip[:, 0].mean(), ip[:, 1].mean()
            lines.append({"n": (nx, ny), "c": -(nx * cx + ny * cy), "centroid": (cx, cy), "w": len(inl)})
        work = work.select_by_index(inl, invert=True)
    return lines

def manhattanize(lines):
    if not lines:
        return lines
    base = max(lines, key=lambda l: l["w"])
    theta0 = math.atan2(base["n"][1], base["n"][0]) % (math.pi / 2)
    for l in lines:
        phi = math.atan2(l["n"][1], l["n"][0])
        snap = theta0 + round((phi - theta0) / (math.pi / 2)) * (math.pi / 2)
        nx, ny = math.cos(snap), math.sin(snap)
        cx, cy = l["centroid"]
        l["n"], l["c"] = (nx, ny), -(nx * cx + ny * cy)
    return lines

def dominant_angle(lines):
    if not lines:
        return 0.0
    base = max(lines, key=lambda l: l["w"])
    return math.atan2(base["n"][1], base["n"][0]) % (math.pi / 2)

def rot(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    return pts @ np.array([[c, -s], [s, c]]).T

def rectilinearize(v):
    n = len(v)
    horiz = [abs(v[(i + 1) % n, 0] - v[i, 0]) >= abs(v[(i + 1) % n, 1] - v[i, 1]) for i in range(n)]
    val = [(v[i, 1] + v[(i + 1) % n, 1]) / 2 if horiz[i] else (v[i, 0] + v[(i + 1) % n, 0]) / 2 for i in range(n)]
    out = []
    for i in range(n):
        p = (i - 1) % n
        if horiz[p] != horiz[i]:
            out.append([val[i] if not horiz[i] else val[p], val[i] if horiz[i] else val[p]])
        else:
            out.append(list(v[i]))
    return np.array(out)

def room_from_floor_mask(floor_pts, ang, r):
    if len(floor_pts) == 0:
        return None
    aligned = rot(floor_pts, -ang)
    mn = aligned.min(0)
    g = np.floor((aligned - mn) / r["mask_res"]).astype(int)
    mask = np.zeros((g[:, 1].max() + 3, g[:, 0].max() + 3), np.uint8)
    mask[g[:, 1] + 1, g[:, 0] + 1] = 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (r["morph_kernel"], r["morph_kernel"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=r["morph_iters"])
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, r["approx_tol"] / r["mask_res"], True).reshape(-1, 2).astype(float)
    if len(approx) < 4:
        return None
    approx = rectilinearize(approx)
    poly = Polygon(rot(approx * r["mask_res"] + mn, ang)).simplify(r["simplify"])
    return poly.buffer(0) if not poly.is_valid else poly

def room_from_floor_obb(floor_pts, ang):
    if len(floor_pts) == 0:
        return None
    rp = rot(floor_pts, -ang)
    lo, hi = np.percentile(rp, 1, 0), np.percentile(rp, 99, 0)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return Polygon(rot(corners, ang))

def detect_openings(room, sub, op):
    coords = list(room.exterior.coords)
    edges = list(zip(coords[:-1], coords[1:]))
    res = []
    for kind in ("door", "window"):
        pts = xy(sub[kind])
        if len(pts) == 0:
            continue
        z = np.asarray(sub[kind].points)[:, 2]
        for p, q in edges:
            p, q = np.array(p), np.array(q)
            v = q - p
            L = math.hypot(*v)
            if L < 1e-6:
                continue
            u = v / L
            d = (pts - p) @ np.array([-u[1], u[0]])
            t = (pts - p) @ u
            on = (np.abs(d) < op["band"]) & (t >= 0) & (t <= L)
            if on.sum() < op["min_points"]:
                continue
            a, b = np.percentile(t[on], [op["pct_lo"], op["pct_hi"]])
            if b - a < op["min_width"]:
                continue
            res.append({"type": kind, "p": (p + a * u).tolist(), "q": (p + b * u).tolist(),
                        "width": float(b - a), "sill": float(z.min()), "head": float(z.max())})
    return res

def detect_columns(col_pcd, room, co):
    if len(col_pcd.points) == 0:
        return []
    labels = np.array(col_pcd.cluster_dbscan(eps=co["eps"], min_points=co["min_points"]))
    pts = np.asarray(col_pcd.points)[:, :2]
    cols = []
    for lb in set(labels):
        if lb < 0:
            continue
        c = pts[labels == lb]
        center = c.mean(0)
        r = float(np.median(np.linalg.norm(c - center, axis=1)))
        if co["r_min"] < r < co["r_max"] and room.contains(Point(center)) and \
                room.exterior.distance(Point(center)) > co["interior_margin"]:
            cols.append({"center": center.tolist(), "radius": r})
    return cols

def z_levels(sub, ceiling_fallback):
    fz = np.asarray(sub["floor"].points)[:, 2]
    cz = np.asarray(sub["ceiling"].points)[:, 2]
    floor = float(np.median(fz)) if len(fz) else 0.0
    ceil = float(np.median(cz)) if len(cz) else floor + ceiling_fallback
    return floor, ceil

def setup_layers(doc):
    spec = [("A-WALL", 7, 50), ("A-WALL-PATT", 8, 9), ("A-DOOR", 7, 25), ("A-GLAZ", 7, 18),
            ("A-COLS", 7, 35), ("A-DIMS", 7, 13), ("A-ANNO", 7, 18), ("A-LEGEND", 7, 18),
            ("A-TTLB", 7, 35), ("A-MASK", 7, 0)]
    for name, color, lw in spec:
        doc.layers.add(name, color=color, lineweight=lw)

def wall_band(msp, ext_mm, inner_mm):
    msp.add_lwpolyline(ext_mm, close=True, dxfattribs={"layer": "A-WALL"})
    if inner_mm is not None:
        msp.add_lwpolyline(inner_mm, close=True, dxfattribs={"layer": "A-WALL"})
    hatch = msp.add_hatch(dxfattribs={"layer": "A-WALL-PATT"})
    hatch.rgb = (110, 110, 110)
    hatch.set_solid_fill()
    hatch.paths.add_polyline_path(ext_mm, is_closed=True, flags=ezdxf.const.BOUNDARY_PATH_EXTERNAL)
    if inner_mm is not None:
        hatch.paths.add_polyline_path(inner_mm, is_closed=True, flags=ezdxf.const.BOUNDARY_PATH_DEFAULT)

def draw_opening(msp, o, cen, th, swing):
    p, q = np.array(o["p"]) * S, np.array(o["q"]) * S
    L = np.linalg.norm(q - p)
    if L < 1:
        return
    u = (q - p) / L
    nrm = np.array([-u[1], u[0]])
    inrm = nrm if np.dot(nrm, cen - (p + q) / 2) > 0 else -nrm
    half = th / 2 + 8
    rect = [p + nrm * half, q + nrm * half, q - nrm * half, p - nrm * half]
    mask = msp.add_hatch(dxfattribs={"layer": "A-MASK"})
    mask.rgb = (255, 255, 255)
    mask.set_solid_fill()
    mask.paths.add_polyline_path([tuple(c) for c in rect], is_closed=True)
    msp.add_line(p + nrm * (th / 2), p - nrm * (th / 2), dxfattribs={"layer": "A-WALL"})
    msp.add_line(q + nrm * (th / 2), q - nrm * (th / 2), dxfattribs={"layer": "A-WALL"})
    if o["type"] == "door":
        hinge = p
        r = min(L, swing)
        msp.add_line(hinge, hinge + inrm * r, dxfattribs={"layer": "A-DOOR"})
        a0 = math.degrees(math.atan2(u[1], u[0]))
        a1 = math.degrees(math.atan2(inrm[1], inrm[0]))
        if (a1 - a0) % 360 > 180:
            a0, a1 = a1, a0
        msp.add_arc(hinge, r, a0, a1, dxfattribs={"layer": "A-DOOR"})
    else:
        for off in (th / 2, 0.0, -th / 2):
            msp.add_line(p + nrm * off, q + nrm * off, dxfattribs={"layer": "A-GLAZ"})

def dim_style(doc, th_txt):
    if "PLAN" not in doc.dimstyles:
        ds = doc.dimstyles.add("PLAN")
        ds.dxf.dimtxt = th_txt
        ds.dxf.dimasz = th_txt * 0.9
        ds.dxf.dimexe = th_txt * 0.5
        ds.dxf.dimexo = th_txt * 0.4
        ds.dxf.dimgap = th_txt * 0.3
        ds.dxf.dimdec = 0

def add_dims(msp, ext_mm, cen, off, th_txt):
    for a, b in zip(ext_mm[:-1], ext_mm[1:]):
        a, b = np.array(a), np.array(b)
        L = np.linalg.norm(b - a)
        if L < 900:
            continue
        u = (b - a) / L
        out = np.array([-u[1], u[0]])
        if np.dot(out, (a + b) / 2 - cen) < 0:
            out = -out
        base = (a + b) / 2 + out * off
        ang = 0 if abs(u[0]) >= abs(u[1]) else 90
        msp.add_linear_dim(base=tuple(base), p1=tuple(a), p2=tuple(b), angle=ang,
                           dimstyle="PLAN", dxfattribs={"layer": "A-DIMS"}).render()

def add_overall_dims(msp, bounds_mm, off, th_txt):
    x0, y0, x1, y1 = bounds_mm
    msp.add_linear_dim(base=(0, y0 - off), p1=(x0, y0), p2=(x1, y0), angle=0,
                       dimstyle="PLAN", dxfattribs={"layer": "A-DIMS"}).render()
    msp.add_linear_dim(base=(x0 - off, 0), p1=(x0, y0), p2=(x0, y1), angle=90,
                       dimstyle="PLAN", dxfattribs={"layer": "A-DIMS"}).render()

def add_scale_bar(msp, x, y, th_txt):
    seg = 1000.0
    bh = th_txt * 0.6
    for i in range(4):
        h = msp.add_hatch(dxfattribs={"layer": "A-ANNO"})
        h.rgb = (0, 0, 0) if i % 2 == 0 else (255, 255, 255)
        h.set_solid_fill()
        h.paths.add_polyline_path([(x + i * seg, y), (x + (i + 1) * seg, y),
                                   (x + (i + 1) * seg, y + bh), (x + i * seg, y + bh)], is_closed=True)
    msp.add_lwpolyline([(x, y), (x + 4 * seg, y), (x + 4 * seg, y + bh), (x, y + bh)],
                       close=True, dxfattribs={"layer": "A-ANNO"})
    for i in range(5):
        msp.add_text(str(i), height=th_txt * 0.7, dxfattribs={"layer": "A-ANNO"}).set_placement(
            (x + i * seg, y - th_txt), align=TextEntityAlignment.CENTER)
    msp.add_text("metres", height=th_txt * 0.7, dxfattribs={"layer": "A-ANNO"}).set_placement(
        (x + 4 * seg + th_txt, y), align=TextEntityAlignment.LEFT)

def add_north(msp, x, y, r):
    msp.add_line((x, y - r), (x, y + r), dxfattribs={"layer": "A-ANNO"})
    for dx in (-r * 0.35, r * 0.35):
        msp.add_line((x + dx, y + r * 0.4), (x, y + r), dxfattribs={"layer": "A-ANNO"})
    msp.add_text("N", height=r * 0.6, dxfattribs={"layer": "A-ANNO"}).set_placement(
        (x, y + r * 1.3), align=TextEntityAlignment.CENTER)

def add_legend(msp, x, y, th_txt):
    dy = th_txt * 2.0
    sw = th_txt * 1.6
    msp.add_text("LEGEND", height=th_txt * 1.1, dxfattribs={"layer": "A-LEGEND"}).set_placement(
        (x, y), align=TextEntityAlignment.LEFT)
    rows = ["WALL", "DOOR", "WINDOW", "COLUMN", "DIMENSION (mm)"]
    for i, label in enumerate(rows):
        ry = y - (i + 1) * dy
        cx = x + sw / 2
        if label == "WALL":
            h = msp.add_hatch(dxfattribs={"layer": "A-LEGEND"})
            h.rgb = (110, 110, 110)
            h.set_solid_fill()
            h.paths.add_polyline_path([(x, ry), (x + sw, ry), (x + sw, ry + sw), (x, ry + sw)], is_closed=True)
        elif label == "DOOR":
            msp.add_line((x, ry), (x, ry + sw), dxfattribs={"layer": "A-LEGEND"})
            msp.add_arc((x, ry), sw, 0, 90, dxfattribs={"layer": "A-LEGEND"})
        elif label == "WINDOW":
            for off in (0.0, sw / 2, sw):
                msp.add_line((x, ry + off), (x + sw, ry + off), dxfattribs={"layer": "A-LEGEND"})
        elif label == "COLUMN":
            msp.add_circle((cx, ry + sw / 2), sw / 2, dxfattribs={"layer": "A-LEGEND"})
        else:
            msp.add_line((x, ry + sw / 2), (x + sw, ry + sw / 2), dxfattribs={"layer": "A-LEGEND"})
        msp.add_text(label, height=th_txt * 0.8, dxfattribs={"layer": "A-LEGEND"}).set_placement(
            (x + sw + th_txt, ry + sw / 2), align=TextEntityAlignment.MIDDLE_LEFT)

def add_title_block(msp, x0, y0, w, h, th_txt, area, scale):
    msp.add_lwpolyline([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)],
                       close=True, dxfattribs={"layer": "A-TTLB"})
    rows = [("PROJECT", "Deep3D-FloorPlan-Net"), ("DRAWING", "Ground Floor Plan"),
            ("SCALE", f"1:{scale}"), ("UNITS", "mm"), ("AREA", f"{area:.1f} m2"),
            ("DATE", datetime.date.today().isoformat()), ("DRAWN", "Deep3D")]
    rh = h / len(rows)
    for i, (k, v) in enumerate(rows):
        ry = y0 + h - (i + 1) * rh
        msp.add_line((x0, ry), (x0 + w, ry), dxfattribs={"layer": "A-TTLB"})
        msp.add_text(k, height=th_txt * 0.7, dxfattribs={"layer": "A-TTLB"}).set_placement(
            (x0 + th_txt * 0.5, ry + rh / 2), align=TextEntityAlignment.MIDDLE_LEFT)
        msp.add_text(v, height=th_txt * 0.8, dxfattribs={"layer": "A-TTLB"}).set_placement(
            (x0 + w * 0.4, ry + rh / 2), align=TextEntityAlignment.MIDDLE_LEFT)

def export_drawing(out_base, room, openings, columns, floor_z, ceil_z, cfg):
    thickness = cfg["wall_thickness"]
    scale = cfg["drawing"]["scale"]
    swing = cfg["drawing"]["door_swing_max"] * S
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    setup_layers(doc)

    ext = np.array(room.exterior.coords) * S
    cen = np.array(room.centroid.coords[0]) * S
    inner = room.buffer(-thickness)
    inner_mm = np.array(inner.exterior.coords) * S if inner.geom_type == "Polygon" and not inner.is_empty else None
    th = thickness * S
    x0, y0, x1, y1 = np.array(room.bounds) * S
    w, h = x1 - x0, y1 - y0
    mw = max(w, h)
    th_txt = max(150.0, mw / 45)
    dim_style(doc, th_txt)

    wall_band(msp, [tuple(c) for c in ext], [tuple(c) for c in inner_mm] if inner_mm is not None else None)
    for o in openings:
        draw_opening(msp, o, cen, th, swing)
    for i, c in enumerate(columns):
        center = (c["center"][0] * S, c["center"][1] * S)
        msp.add_circle(center, c["radius"] * S, dxfattribs={"layer": "A-COLS"})
        msp.add_text(f"C{i + 1}", height=th_txt * 0.7, dxfattribs={"layer": "A-COLS"}).set_placement(
            center, align=TextEntityAlignment.MIDDLE_CENTER)

    add_dims(msp, [tuple(c) for c in ext], cen, th + th_txt * 1.5, th_txt)
    overall_off = th + th_txt * 5.0
    add_overall_dims(msp, (x0, y0, x1, y1), overall_off, th_txt)

    msp.add_mtext(f"ROOM\\P{room.area:.1f} m2", dxfattribs={
        "layer": "A-ANNO", "char_height": th_txt, "attachment_point": 5}).set_location(tuple(cen))

    sb_y = y0 - (th + th_txt * 8.5)
    add_scale_bar(msp, x0, sb_y, th_txt)
    add_north(msp, x1 + mw * 0.10, y1 + th_txt * 2, th_txt * 1.6)
    lx = x1 + mw * 0.20
    add_legend(msp, lx, y1, th_txt)
    tb_w, tb_h = mw * 0.5, th_txt * 11.0
    add_title_block(msp, lx, y0, tb_w, tb_h, th_txt, room.area, scale)

    fx0 = x0 - overall_off - th_txt * 3
    fx1 = lx + tb_w + th_txt * 3
    fy0 = sb_y - th_txt * 3
    fy1 = y1 + th_txt * 6
    msp.add_lwpolyline([(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)], close=True, dxfattribs={"layer": "A-TTLB"})

    doc.saveas(out_base + ".dxf")
    for ext_name in (".png", ".pdf", ".svg"):
        dxf_mpl.qsave(msp, out_base + ext_name, dpi=200, bg="#FFFFFF")

def export_ifc(path, room, openings, columns, floor_z, ceil_z, thickness):
    import ifcopenshell
    from ifcopenshell.api import run
    f = ifcopenshell.file(schema="IFC4")
    units = [f.create_entity("IfcSIUnit", UnitType=u, Name=n) for u, n in
             [("LENGTHUNIT", "METRE"), ("AREAUNIT", "SQUARE_METRE"), ("VOLUMEUNIT", "CUBIC_METRE")]]
    project = run("root.create_entity", f, ifc_class="IfcProject", name="Deep3D-FloorPlan")
    run("unit.assign_unit", f, units=units)
    ctx = run("context.add_context", f, context_type="Model")
    body = run("context.add_context", f, context_type="Model", context_identifier="Body",
               target_view="MODEL_VIEW", parent=ctx)
    site = run("root.create_entity", f, ifc_class="IfcSite", name="Site")
    bldg = run("root.create_entity", f, ifc_class="IfcBuilding", name="Building")
    storey = run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Storey")
    run("aggregate.assign_object", f, product=site, relating_object=project)
    run("aggregate.assign_object", f, product=bldg, relating_object=site)
    run("aggregate.assign_object", f, product=storey, relating_object=bldg)

    h = ceil_z - floor_z
    coords = list(room.exterior.coords)[:-1]
    edges = [(coords[i], coords[(i + 1) % len(coords)]) for i in range(len(coords))]
    walls = []
    for i, (p1, p2) in enumerate(edges):
        p1a, p2a = np.array(p1), np.array(p2)
        L = float(np.linalg.norm(p2a - p1a))
        if L < 1e-3:
            walls.append(None)
            continue
        wdir = (p2a - p1a) / L
        mid = (p1a + p2a) / 2
        wall = run("root.create_entity", f, ifc_class="IfcWall", name=f"Wall-{i}")
        wall.Representation = _box(f, body, L, thickness, h)
        wall.ObjectPlacement = _placement(f, (mid[0], mid[1], floor_z), xdir=wdir)
        run("spatial.assign_container", f, product=wall, relating_structure=storey)
        walls.append(wall)

    for name, z, depth in [("Floor", floor_z - 0.15, 0.15), ("Ceiling", ceil_z, 0.15)]:
        slab = run("root.create_entity", f, ifc_class="IfcSlab", name=name)
        slab.Representation = _shape(f, body, _extrude(f, _poly_profile(f, room.exterior), depth))
        slab.ObjectPlacement = _placement(f, (0, 0, z))
        run("spatial.assign_container", f, product=slab, relating_structure=storey)

    space = run("root.create_entity", f, ifc_class="IfcSpace", name="Room")
    run("aggregate.assign_object", f, product=space, relating_object=storey)

    for o in openings:
        p, q = np.array(o["p"]), np.array(o["q"])
        mid = (p + q) / 2
        wdir = (q - p) / (np.linalg.norm(q - p) + 1e-9)
        w = float(np.linalg.norm(q - p))
        oh = o["head"] - o["sill"]
        wall = walls[_nearest_edge(mid, edges)]
        if wall is None:
            continue
        opening = run("root.create_entity", f, ifc_class="IfcOpeningElement", name="Opening")
        opening.Representation = _box(f, body, w, thickness + 0.1, oh)
        opening.ObjectPlacement = _placement(f, (mid[0], mid[1], o["sill"]), xdir=wdir)
        run("void.add_opening", f, opening=opening, element=wall)
        cls = "IfcDoor" if o["type"] == "door" else "IfcWindow"
        fill = run("root.create_entity", f, ifc_class=cls, name=o["type"].capitalize())
        fill.Representation = _box(f, body, w, thickness, oh)
        fill.ObjectPlacement = _placement(f, (mid[0], mid[1], o["sill"]), xdir=wdir)
        run("void.add_filling", f, opening=opening, element=fill)

    for c in columns:
        col = run("root.create_entity", f, ifc_class="IfcColumn", name="Column")
        prof = f.create_entity("IfcCircleProfileDef", ProfileType="AREA",
                               Position=f.create_entity("IfcAxis2Placement2D", Location=_pt(f, (0., 0.))),
                               Radius=float(c["radius"]))
        col.Representation = _shape(f, body, _extrude(f, prof, h))
        col.ObjectPlacement = _placement(f, (c["center"][0], c["center"][1], floor_z))
        run("spatial.assign_container", f, product=col, relating_structure=storey)

    f.write(path)

def _pt(f, c):
    return f.create_entity("IfcCartesianPoint", Coordinates=tuple(float(v) for v in c))

def _placement(f, origin=(0, 0, 0), xdir=None):
    a = {"Location": _pt(f, origin)}
    if xdir is not None:
        a["Axis"] = f.create_entity("IfcDirection", DirectionRatios=(0., 0., 1.))
        a["RefDirection"] = f.create_entity("IfcDirection", DirectionRatios=(float(xdir[0]), float(xdir[1]), 0.))
    return f.create_entity("IfcLocalPlacement", RelativePlacement=f.create_entity("IfcAxis2Placement3D", **a))

def _shape(f, body, solid):
    rep = f.create_entity("IfcShapeRepresentation", ContextOfItems=body,
                          RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[solid])
    return f.create_entity("IfcProductDefinitionShape", Representations=[rep])

def _extrude(f, profile, depth):
    return f.create_entity("IfcExtrudedAreaSolid", SweptArea=profile,
                           Position=f.create_entity("IfcAxis2Placement3D", Location=_pt(f, (0., 0., 0.))),
                           ExtrudedDirection=f.create_entity("IfcDirection", DirectionRatios=(0., 0., 1.)),
                           Depth=float(depth))

def _poly_profile(f, ring):
    pts = [_pt(f, (x, y)) for x, y in ring.coords]
    return f.create_entity("IfcArbitraryClosedProfileDef", ProfileType="AREA",
                           OuterCurve=f.create_entity("IfcPolyline", Points=pts))

def _box(f, body, w, d, h):
    prof = f.create_entity("IfcRectangleProfileDef", ProfileType="AREA",
                           Position=f.create_entity("IfcAxis2Placement2D", Location=_pt(f, (0., 0.))),
                           XDim=float(w), YDim=float(d))
    return _shape(f, body, _extrude(f, prof, h))

def _nearest_edge(mid, edges):
    best, bd = 0, 1e9
    for i, (p1, p2) in enumerate(edges):
        p1, p2 = np.array(p1), np.array(p2)
        v = p2 - p1
        t = np.clip((np.array(mid) - p1) @ v / (v @ v + 1e-9), 0, 1)
        d = np.linalg.norm(p1 + t * v - mid)
        if d < bd:
            best, bd = i, d
    return best

def reconstruct(clouds, cfg):
    boundary = clouds["walls"] + clouds["window"] + clouds["door"]
    lines = manhattanize(vertical_wall_lines(boundary, cfg["walls"]))
    ang = dominant_angle(lines)
    floor_pts = xy(clouds["floor"])
    fbb = (floor_pts.max(0) - floor_pts.min(0)) if len(floor_pts) else np.zeros(2)
    area = float(fbb[0] * fbb[1])
    method = cfg["method"]
    room = None if method == "obb" else room_from_floor_mask(floor_pts, ang, cfg["room"])
    used = "floor_mask"
    bad = room is None or not (cfg["room"]["area_lo"] * area <= room.area <= cfg["room"]["area_hi"] * area)
    if room is None or (bad and method != "floor_mask"):
        room, used = room_from_floor_obb(floor_pts, ang), "obb"
    openings = detect_openings(room, clouds, cfg["openings"])
    columns = detect_columns(clouds["column"], room, cfg["columns"])
    floor_z, ceil_z = z_levels(clouds, cfg["ceiling_fallback_height"])
    return room, openings, columns, floor_z, ceil_z, used

def generate(clouds, out_dir, cfg):
    os.makedirs(out_dir, exist_ok=True)
    room, openings, columns, floor_z, ceil_z, method = reconstruct(clouds, cfg)
    export_drawing(os.path.join(out_dir, "floor_plan"), room, openings, columns, floor_z, ceil_z, cfg)
    export_ifc(os.path.join(out_dir, "model.ifc"), room, openings, columns, floor_z, ceil_z, cfg["wall_thickness"])
    info = {"method": method, "area_m2": round(room.area, 3),
            "bounds": [round(v, 3) for v in room.bounds],
            "floor_z": round(floor_z, 3), "ceil_z": round(ceil_z, 3),
            "n_openings": len(openings), "n_columns": len(columns),
            "openings": openings, "columns": columns}
    json.dump(info, open(os.path.join(out_dir, "floorplan.json"), "w"), indent=2)
    return info

def build(seg_dir, out_dir, cfg):
    return generate(load_structural(seg_dir), out_dir, cfg)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SOTA Stage-3 floor plan generation")
    ap.add_argument("--seg_dir", default="output/segmented_parts")
    ap.add_argument("--out_dir", default="output/sota")
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--method", choices=["auto", "floor_mask", "obb"])
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.method:
        cfg["method"] = a.method
    print(json.dumps(build(a.seg_dir, a.out_dir, cfg), indent=2)[:800])
