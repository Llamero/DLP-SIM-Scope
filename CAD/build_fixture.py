"""
DLP illumination fixture  --  concept v1
Generates a single rigid PerFORM (SLA) fixture in the SAME GLOBAL COORDINATES as DLP_Scope.FCStd,
so the STEP imports on top of your optics with no placement.

Run:  python build_fixture.py        (needs: pip install cadquery)
Out:  DLP_fixture_v1.step / .stl

Every number marked [FILE] was measured from DLP_Scope.FCStd.
Every number marked [ASSUME] is a guess you must confirm before printing.
"""
import math
import numpy as np
import cadquery as cq
from cadquery import Vector, Solid

# ----------------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------------
# --- DMD / EVM -------------------------------------------------------------------
DMD_C        = np.array([-0.70, 0.30, -100.40])   # [FILE] centre of DMD cavity / window (x,y,z)
BOARD_FACE_Z = -105.60                            # [FILE] DMD-side face of the EVM PCB
PILLOW_HALF  = 21.1                               # [FILE] DMD ceramic package half-width (42.2 sq)
POST_XY      = {"A": (-30.7, -3.3), "B": (29.3, -3.3), "D": (-0.7, 30.8)}   # [FILE] 3 of the 4 x3.5 mm PCB holes
SHIM_GAP     = 0.30                               # nominal shim-pack thickness under each DMD post / mirror land
COLLAR_Z     = (-101.5, -98.0)                    # collar ring z-range (ceramic top is -97.48)
COLLAR_OUT   = 35.0                               # collar outer half-width about DMD_C
COLLAR_IN    = 23.6                               # collar opening half-width (pillow 21.1 + 2.5)
POST_D       = 9.0
INSERT_M3_D  = 4.2                                # heat-set insert pilot (M3), check your insert datasheet

# --- Static achromat (lens 1) ---------------------------------------------------------
LENS1_Z      = (-10.50, 4.00)                     # [FILE] edge-to-edge z of doublet 1
LENS1_OD     = 50.0                               # [ASSUME] model shows 49.0; real 2" lenses are usually 50.0
LENS_CLR     = 0.40                               # diametral clearance in bore
LENS1_LEDGE_IN = 23.3                             # seat ledge inner radius (lens CA is smaller than OD)
RING_OUT_R   = 33.0
RING_Z       = (-12.0, 4.6)                       # lens 2 starts at z = +5.0 -> keep ring below it
RING_OPEN_HALF_DEG = 42.0                         # sector cut open toward mirror (F-frame azimuth 0 = +s)

# --- Mirror (1" 90 deg OAP) ------------------------------------------------------
CB  = np.array([29.351, -28.823, -9.614])         # [FILE] centre of mirror back face
NB  = np.array([0.2876, -0.2876, 0.9135])         # [FILE] back-face normal (points away from DMD)
NB /= np.linalg.norm(NB)
EX  = np.array([-0.7071, -0.7071, 0.0])           # in-plane axis u
EY  = np.cross(NB, EX)                            # in-plane axis v
TAP_UV    = [(-9.3, -2.3), (2.3, 9.3), (6.6, -6.6)]   # [FILE] 3x 4-40 (tap drill 2.26, 6.35 deep)
DOWEL_ROUND = (-0.1, 0.1)                         # [FILE] dowel hole #1 (3.19 dia, 3.2 deep)
DOWEL_SLOT  = (-6.9, 6.9)                         # [FILE] dowel hole #2 -> made a slot (exact constraint)
MIRROR_R     = 12.7
MIRROR_LEN   = 31.5                               # [FILE] shell length behind back face
MIRROR_CLR   = 1.0

# --- Asphere + LLG -----------------------------------------------------------------
LLG_FACE = np.array([50.08, -22.16, -34.95])      # [FILE] LLG output face centre
LLG_DIR  = np.array([-0.957, -0.043, 0.288])      # [FILE] beam direction (LLG -> asphere -> OAP)
LLG_DIR /= np.linalg.norm(LLG_DIR)
ASPH_R         = 12.7                             # [FILE] 1" asphere
ASPH_SEAT_A    = 7.80                             # [FILE] first-surface height at r = 11.0 (circular line seat)
ASPH_SEAT_R    = 11.0
ASPH_BORE_R    = 13.0                             # 12.7 + 0.3
ASPH_RING_R    = 17.0
ASPH_FLOOR_A   = (3.4, 7.0)
ASPH_WALL_A_TOP = 9.8
CONE_NA        = 0.59
FERRULE_D      = 6.0                              # [ASSUME] LLG output ferrule OD  -- MEASURE
BOSS_OD        = 16.0
BOSS_A         = (-24.0, -3.0)

# --- Wall / spine (F-frame: s along (+x,-y)/sqrt2, t along (+x,+y)/sqrt2) ------------
WALL_T   = (27.0, 33.0)
WALL_S   = (-10.0, 76.0)
WALL_Z   = (-101.5, 4.6)

# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def V(a): return Vector(float(a[0]), float(a[1]), float(a[2]))

def boxF(s0, s1, t0, t1, z0, z1):
    """box in the 45-deg-rotated (s,t,z) frame -> global"""
    b = cq.Workplane("XY").box(s1 - s0, t1 - t0, z1 - z0, centered=False).translate((s0, t0, z0))
    return b.rotate((0, 0, 0), (0, 0, 1), -45).val()

def boxXY(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0)).val()

def cyl(p0, d, r, L):
    return Solid.makeCylinder(r, L, V(p0), V(d))

def cone(p0, d, r0, r1, L):
    return Solid.makeCone(r0, r1, L, V(p0), V(d))

def axpt(origin, d, a):
    return np.asarray(origin) + a * np.asarray(d)

def fuse_all(shapes):
    out = shapes[0]
    for s in shapes[1:]:
        out = out.fuse(s)
    return out.clean()

# mirror-frame solids -------------------------------------------------------------
MIR_PLANE = cq.Plane(origin=tuple(CB), xDir=tuple(EX), normal=tuple(NB))
def mir_box(u0, u1, v0, v1, n0, n1):
    s = Solid.makeBox(u1 - u0, v1 - v0, n1 - n0, pnt=Vector(u0, v0, n0))
    return s.moved(cq.Location(MIR_PLANE))
def mir_cyl(u, v, n0, n1, r):
    s = Solid.makeCylinder(r, n1 - n0, Vector(u, v, n0), Vector(0, 0, 1))
    return s.moved(cq.Location(MIR_PLANE))

# LLG-axis frame ---------------------------------------------------------------------
def llg_cyl(a0, a1, r):
    return cyl(axpt(LLG_FACE, LLG_DIR, a0), LLG_DIR, r, a1 - a0)

# ----------------------------------------------------------------------------------
# KEEP-OUT VOLUMES (subtracted from fixture)
# ----------------------------------------------------------------------------------
def keepouts():
    K = {}
    # lens 1 + bore handled in ring construction
    # ON-state beam DMD -> lens 1
    K["on_beam"] = cone((0, 0, DMD_C[2]), (0, 0, 1), 9.5, 21.5, LENS1_Z[0] - DMD_C[2])
    # illumination cone mirror -> DMD
    M0 = np.array([23.99, -23.50, -26.54])        # [FILE] mirror optical-surface centroid
    d = DMD_C - M0; L = np.linalg.norm(d); d /= L
    K["illum"] = cone(M0 - 2 * d, d, 13.0, 10.5, L + 2)
    # collimated beam asphere -> OAP
    a_mir = float(np.dot(M0 - LLG_FACE, LLG_DIR))
    K["colli"] = llg_cyl(9.8, a_mir, 10.8)
    # LLG divergence cone
    K["llg_cone"] = cone(LLG_FACE - 0.5 * LLG_DIR, LLG_DIR, 1.5, 7.3, 7.8)
    # OFF-state beam (48 deg from normal toward -s)
    es = np.array([0.7071, -0.7071, 0.0])
    off = -math.sin(math.radians(48)) * es + math.cos(math.radians(48)) * np.array([0, 0, 1.0])
    K["off_beam"] = cone(DMD_C, off, 9.5, 26.0, 125.0)
    # mirror body (conservative cylinder), only behind->front side (n <= 0)
    K["mirror"] = mir_cyl(0, 0, -MIRROR_LEN - 1.5, 0.0, MIRROR_R + MIRROR_CLR)
    # asphere dome (a >= 9.8) : revolve profile from measured max radius, +0.8 clearance
    prof = [(9.8, 13.5), (11.0, 12.7), (13.0, 11.7), (15.0, 10.5), (17.0, 9.2), (19.0, 7.1), (21.0, 3.6), (21.6, 0.0)]
    K["asph_dome"] = fuse_all([
        cone(axpt(LLG_FACE, LLG_DIR, a0), LLG_DIR, r0, r1, a1 - a0)
        for (a0, r0), (a1, r1) in zip(prof[:-1], prof[1:])
    ])
    # lens 1 / lens 2 envelopes
    K["lens1"] = cyl((0, 0, LENS1_Z[0] - 0.2), (0, 0, 1), LENS1_OD / 2 + 0.2, LENS1_Z[1] - LENS1_Z[0] + 0.4)
    # DMD ceramic pillow
    K["pillow"] = boxXY(DMD_C[0] - PILLOW_HALF - 0.3, DMD_C[0] + PILLOW_HALF + 0.3,
                        DMD_C[1] - PILLOW_HALF - 0.3, DMD_C[1] + PILLOW_HALF + 0.3,
                        -104.6, -93.0)
    return K

# ----------------------------------------------------------------------------------
# FIXTURE
# ----------------------------------------------------------------------------------
def build():
    print("keep outs")
    K = keepouts()
    parts = {}

    # 1. spine wall -----------------------------------------------------------------
    print("spine")
    parts["wall"] = boxF(WALL_S[0], WALL_S[1], WALL_T[0], WALL_T[1], WALL_Z[0], WALL_Z[1])

    # 2. DMD collar + 3 posts -----------------------------------------------------------
    print("DMD")
    cx, cy = DMD_C[0], DMD_C[1]
    collar = boxXY(cx - COLLAR_OUT, cx + COLLAR_OUT, cy - COLLAR_OUT, cy + COLLAR_OUT, COLLAR_Z[0], COLLAR_Z[1])
    collar = collar.cut(boxXY(cx - COLLAR_IN, cx + COLLAR_IN, cy - COLLAR_IN, cy + COLLAR_IN, COLLAR_Z[0] - 1, COLLAR_Z[1] + 1))
    parts["collar"] = collar
    post_z0 = BOARD_FACE_Z + SHIM_GAP
    posts = []
    for k, (x, y) in POST_XY.items():
        posts.append(cyl((x, y, post_z0), (0, 0, 1), POST_D / 2, COLLAR_Z[0] - post_z0 + 0.5))
    parts["posts"] = fuse_all(posts)

    # 3. lens-1 ring (C-shape, open toward mirror) -----------------------------------------
    print("lens ring")
    ring = cyl((0, 0, RING_Z[0]), (0, 0, 1), RING_OUT_R, RING_Z[1] - RING_Z[0])
    # sector open toward the mirror (azimuth 0 deg in F-frame = +s)
    sector = boxF(0, 60, -60 * math.tan(math.radians(RING_OPEN_HALF_DEG)) * 0.0 - 45, 45, RING_Z[0] - 1, RING_Z[1] + 1)
    wedge = (cq.Workplane("XY").polyline([(0, 0),
                (80 * math.cos(math.radians(RING_OPEN_HALF_DEG)), 80 * math.sin(math.radians(RING_OPEN_HALF_DEG))),
                (80 * math.cos(math.radians(RING_OPEN_HALF_DEG)), -80 * math.sin(math.radians(RING_OPEN_HALF_DEG)))]).close()
             .extrude(RING_Z[1] - RING_Z[0] + 2).translate((0, 0, RING_Z[0] - 1)).rotate((0, 0, 0), (0, 0, 1), -45).val())
    ring = ring.cut(wedge)
    parts["ring"] = ring

    # 4. mirror bridge slab ------------------------------------------------------------
    print("mirror bridge")
    u_wall = 0.37 - WALL_T[1]                       # t = 0.37 - u  (EX = -e_t)
    slab = mir_box(u_wall, 16.0, -10.0, 16.0, 0.8, 10.0)
    for (u, v) in TAP_UV:                           # raised lands (shim seats)
        slab = slab.fuse(mir_cyl(u, v, SHIM_GAP, 0.9, 3.5))
    parts["slab"] = slab.clean()

    # 5. asphere cell -------------------------------------------------------------------
    print("asphere")
    cell = []
    cell.append(llg_cyl(ASPH_FLOOR_A[0], ASPH_WALL_A_TOP, ASPH_RING_R))
    cell = fuse_all(cell)
    parts["cell"] = cell

    # 6. ferrule boss ---------------------------------------------------------------------
    print("ferrule")
    parts["boss"] = llg_cyl(BOSS_A[0], BOSS_A[1], BOSS_OD / 2)

    # ---- union ------------------------------------------------------------------------------
    print("union")
    fx = fuse_all(list(parts.values()))

    # ---- subtract keep-outs / bores ----------------------------------------------------------------
    for name in ["on_beam", "illum", "colli", "off_beam", "mirror", "asph_dome", "pillow"]:
        print("Subtract keepouts " + str(name))
        fx = fx.cut(K[name])
    # lens-1 bore + ledge
    print("lens bore")
    fx = fx.cut(cyl((0, 0, RING_Z[0] + 1.5), (0, 0, 1), LENS1_OD / 2 + LENS_CLR / 2, 40))
    fx = fx.cut(cyl((0, 0, RING_Z[0] - 1), (0, 0, 1), LENS1_LEDGE_IN, 5))
    # asphere bore, floor window, seat ring
    fx = fx.cut(llg_cyl(ASPH_FLOOR_A[1], ASPH_WALL_A_TOP + 1.0, ASPH_BORE_R))
    fx = fx.cut(llg_cyl(ASPH_FLOOR_A[0] - 1, ASPH_FLOOR_A[1] + 1, 7.8))
    seat = llg_cyl(ASPH_FLOOR_A[1] - 0.1, ASPH_SEAT_A, ASPH_SEAT_R + 1.2).cut(llg_cyl(ASPH_FLOOR_A[1] - 1, ASPH_SEAT_A + 1, ASPH_SEAT_R))
    fx = fx.fuse(seat)
    # ferrule bore (sliding fit) through boss + wall, stops short of the floor
    print("ferrule bore")
    fx = fx.cut(llg_cyl(-70.0, 1.9, FERRULE_D / 2 + 0.15))
    # collar / posts: M3 insert holes from below, through-clear for the lens-1 + mirror features
    print("Collar and M3 inserts")
    for k, (x, y) in POST_XY.items():
        fx = fx.cut(cyl((x, y, post_z0 - 1), (0, 0, 1), INSERT_M3_D / 2, 7.0))
    # mirror seat: 4-40 clearance + c'bore, dowel round + slot
    print("Mirror 4-40 seat")
    for (u, v) in TAP_UV:
        fx = fx.cut(mir_cyl(u, v, -1, 12, 1.85))
        fx = fx.cut(mir_cyl(u, v, 5.0, 12, 3.5))
    fx = fx.cut(mir_cyl(DOWEL_ROUND[0], DOWEL_ROUND[1], -1, 12, 1.55))
    p0 = np.array(DOWEL_SLOT); dvec = (np.array(DOWEL_SLOT) - np.array(DOWEL_ROUND)); dvec /= np.linalg.norm(dvec)
    for off in (-0.45, 0.45):
        q = p0 + off * dvec
        fx = fx.cut(mir_cyl(q[0], q[1], -1, 12, 1.55))
    slot_mid = p0
    slot_box = Solid.makeBox(0.9, 3.1, 13, pnt=Vector(-0.45, -1.55, -1)).moved(cq.Location(cq.Plane(origin=(0, 0, 0), xDir=(1, 0, 0), normal=(0, 0, 1))))
    # (box oriented along dvec)
    print("box")
    ang = math.degrees(math.atan2(dvec[1], dvec[0]))
    slot_box = slot_box.rotate(Vector(0, 0, 0), Vector(0, 0, 1), ang - 0.0).moved(cq.Location(Vector(slot_mid[0], slot_mid[1], 0)))
    fx = fx.cut(slot_box.moved(cq.Location(MIR_PLANE)))
    # asphere clamp inserts (M2) on wall top, 3x at 120 deg
    print("asphere clamp")
    ang0 = 0.0
    up = np.array([0, 0, 1.0]) - LLG_DIR[2] * LLG_DIR
    up /= np.linalg.norm(up); side = np.cross(LLG_DIR, up)
    for k in range(3):
        th = math.radians(60 + 120 * k)
        rdir = math.cos(th) * up + math.sin(th) * side
        p = axpt(LLG_FACE, LLG_DIR, ASPH_WALL_A_TOP + 0.01) + 15.0 * rdir
        fx = fx.cut(cyl(p - 4.0 * LLG_DIR, LLG_DIR, 1.6, 4.5))
    # lens-1 clamp inserts (M3), 3x on the ring top
    print("lens clamp")
    for phi in (110, 180, 250):
        x = 29.4 * math.cos(math.radians(phi)); y = 29.4 * math.sin(math.radians(phi))
        pt = cq.Workplane("XY").newObject([cq.Vector(x, y, 0)]).rotate((0, 0, 0), (0, 0, 1), -45).val()
        fx = fx.cut(cyl((pt.x, pt.y, RING_Z[1] - 6.0), (0, 0, 1), INSERT_M3_D / 2, 6.5))
    return fx.clean(), parts, K


if __name__ == "__main__":
    print("start")
    fx, parts, K = build()
    print("solids:", len(fx.Solids()), " volume mm3: %.0f" % fx.Volume())
    bb = fx.BoundingBox()
    print("bbox x[%.1f,%.1f] y[%.1f,%.1f] z[%.1f,%.1f]" % (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax))
    cq.exporters.export(cq.Workplane("XY").newObject([fx]), "DLP_fixture_v1.step")
    cq.exporters.export(cq.Workplane("XY").newObject([fx]), "DLP_fixture_v1.stl", tolerance=0.05)
