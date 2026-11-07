

#!/usr/bin/env python3
"""
SIM17.4 — Parent Projectors, Bilinear Blocks,
Arithmetic Admissibility, and Half-Turn Selection

VALIDATOR RUN-BUTTON EDITION

Gates:
 A. 16D parent and Clifford volume identities
 B. Projection selection and parent symmetries
 C. Four bilinear input-sector blocks
 D. Associator projectors and composed 4D projections
 E. Independently specified octavian arithmetic controls
 F. Candidate half-turn selection and symmetry obstruction
 G. Held-out frozen OPS acceptance

Exact integer arithmetic.
No ISP, LCO dynamics, stochastic evolution,
graph fitting, or renormalization.

Dependency: numpy

All conclusions are restricted to declared inputs,
coordinate constructions, and tested symmetry groups.
"""

import itertools as it
import json
from collections import Counter

import numpy as np


# ============================================================
# VALIDATOR UTILITIES
# ============================================================

def gate(label, condition):
    if not bool(condition):
        raise AssertionError("FAIL | " + label)
    print("PASS | " + label)


def cd(i, j, n=16):
    """Exact Cayley-Dickson signed basis multiplication."""
    if n == 1:
        return (1, 0)

    k = n // 2

    if i < k and j < k:
        return cd(i, j, k)

    if i < k and j >= k:
        s, t = cd(j-k, i, k)
        return (s, t+k)

    if i >= k and j < k:
        s, t = cd(i-k, j, k)
        return (s * (1 if j == 0 else -1), t+k)

    s, t = cd(j-k, i-k, k)
    return (-s * (1 if j-k == 0 else -1), t)


SG = np.zeros((16, 16), dtype=np.int64)
IX = np.zeros((16, 16), dtype=np.int64)

for i, j in it.product(range(16), repeat=2):
    SG[i, j], IX[i, j] = cd(i, j)

I16 = np.eye(16, dtype=np.int64)
I8 = np.eye(8, dtype=np.int64)


def left(i):
    out = np.zeros((16, 16), dtype=np.int64)

    for j in range(16):
        out[int(IX[i, j]), j] = SG[i, j]

    return out


J = [left(i) for i in range(16)]


def mul(v, w):
    out = np.zeros(16, dtype=np.int64)

    for i in np.flatnonzero(v):
        for j in np.flatnonzero(w):
            out[int(IX[i, j])] += (
                int(v[i])
                * int(w[j])
                * int(SG[i, j])
            )

    return out


print("=" * 64)
print("SIM17.4 — PARENT PROJECTION / BILINEAR / HALF-TURN AUDIT")
print("=" * 64)


# ============================================================
# GATE A — PARENT AND CLIFFORD VOLUME
# ============================================================

print("\n--- GATE A: PARENT AND CLIFFORD VOLUME ---")

gate(
    "A signed multiplication and 16D identity",
    all(
        int(IX[i, j]) == (i ^ j)
        for i, j in it.product(range(16), repeat=2)
    )
    and all(
        SG[0, i] == SG[i, 0] == 1
        for i in range(16)
    )
)

gate(
    "A individual 15 imaginary orthogonal complex structures",
    all(
        np.array_equal(J[i] @ J[i], -I16)
        and np.array_equal(J[i].T, -J[i])
        for i in range(1, 16)
    )
)

gate(
    "A full 7-generator parent anticommutation",
    all(
        not np.any(J[i] @ J[j] + J[j] @ J[i])
        for i, j in it.combinations(range(1, 8), 2)
    )
)

Gamma7 = I16.copy()
Gamma8 = I16.copy()

for i in range(1, 8):
    Gamma7 = Gamma7 @ J[i]

for i in range(8, 16):
    Gamma8 = Gamma8 @ J[i]

expected = np.diag([-1]*8 + [1]*8)

gate(
    "A Gamma7 = diag(-I8,+I8)",
    np.array_equal(Gamma7, expected)
)

gate(
    "A Gamma8 = -Gamma7",
    np.array_equal(Gamma8, -Gamma7)
)

# Exact integer spectral projectors.
P = (I16 + Gamma8) // 2
K = I16 - P

W = tuple(range(8))
U = tuple(range(8, 16))

gate(
    "A exact complementary orthogonal spectral projectors",
    np.array_equal(P @ P, P)
    and np.array_equal(K @ K, K)
    and np.array_equal(
        P @ K,
        np.zeros((16, 16), dtype=np.int64)
    )
    and np.array_equal(
        P,
        np.diag([1]*8 + [0]*8)
    )
)

print("A | rank P = 8, rank K = 8")


# ============================================================
# GATE B — PROJECTOR SELECTION AND INPUT SYMMETRY
# ============================================================

print("\n--- GATE B: PROJECTOR SELECTION AND INPUT SYMMETRY ---")

# The Clifford families are supplied by the chosen
# Cayley-Dickson splitting. This is not an intrinsic
# selection of the parent decomposition.

# Independently enumerate unital coordinate eight-planes.
# Frozen OPS information is not used.

closed = []

for rest in it.combinations(range(1, 16), 7):
    S = (0,) + rest

    if all(
        int(IX[i, j]) in S
        for i, j in it.product(S, repeat=2)
    ):
        closed.append(S)

ambient = []

for S in closed:
    count = sum(
        not np.any(
            J[i] @ J[j] + J[j] @ J[i]
        )
        for i, j in it.combinations(S[1:], 2)
    )
    ambient.append(count)

gate(
    "B 15 closed coordinate 8-planes",
    len(closed) == 15
)

gate(
    "B only standard W has 21 ambient anticommuting pairs",
    ambient.count(21) == 1
    and closed[ambient.index(21)] == W
)


# ------------------------------------------------------------
# B2 — Signed octonion basis automorphisms
# ------------------------------------------------------------

# These preserve the original octonionic multiplication.
# They can be extended to the parent by
#
#       g(a,b) = (g(a),g(b)).
#
# This constructs a certified parent automorphism SUBGROUP.
# It does not enumerate every sedenion automorphism.

autos = []

for a, b, c in it.product(range(1, 8), repeat=3):

    if b == a or c in (a, b, a ^ b):
        continue

    perm = tuple(
        (a if i & 1 else 0)
        ^ (b if i & 2 else 0)
        ^ (c if i & 4 else 0)
        for i in range(8)
    )

    for mask in range(128):

        signs = (1,) + tuple(
            -1 if mask & (1 << (i-1)) else 1
            for i in range(1, 8)
        )

        if all(
            int(SG[i, j]) * signs[i ^ j]
            ==
            signs[i] * signs[j]
            * int(SG[perm[i], perm[j]])
            for i, j in it.product(range(8), repeat=2)
        ):
            autos.append((perm, signs))

gate(
    "B 1344 signed octonion basis automorphisms",
    len(autos) == 1344
)


# ------------------------------------------------------------
# B3 — Verify each symmetry extends to the 16D parent
# ------------------------------------------------------------

for perm, signs in autos:

    def ap(i):
        return perm[i & 7] + (i & 8)

    def sig(i):
        return signs[i & 7]

    if not all(
        int(SG[i, j]) * sig(i ^ j)
        ==
        sig(i) * sig(j)
        * int(SG[ap(i), ap(j)])
        and int(IX[ap(i), ap(j)]) == ap(i ^ j)
        for i, j in it.product(range(16), repeat=2)
    ):
        raise AssertionError(
            "FAIL | B parent automorphism lift"
        )

gate(
    "B all 1344 octonion automorphisms lift and fix P",
    True
)

orbits = {
    i: {
        perm[i]
        for perm, _ in autos
    }
    for i in range(1, 8)
}

gate(
    "B residual parent symmetry transitive on seven directions",
    all(
        orbit == set(range(1, 8))
        for orbit in orbits.values()
    )
)

print("B | certified parent automorphism subgroup:", len(autos))


# ============================================================
# GATE C — FOUR BILINEAR SECTOR BLOCKS
# ============================================================

print("\n--- GATE C: FOUR BILINEAR SECTOR BLOCKS ---")

# Parent multiplication is bilinear:
#
#   Phi(x,y) = P(xy)
#
# Each input has a component in W or U.
#
# Four tensor blocks:
#   W x W
#   W x U
#   U x W
#   U x U
#
# A rank here is the rank of the tensor-linearized
# bilinear map. Each basis product has one signed
# output coordinate, making this an exact rank count.

def block_rank(rows, cols, output_indices):

    return len({
        int(IX[i, j])
        for i, j in it.product(rows, cols)
        if int(IX[i, j]) in output_indices
    })


block_ranks = {
    name: block_rank(a, b, W)
    for name, a, b in [
        ("WW", W, W),
        ("WU", W, U),
        ("UW", U, W),
        ("UU", U, U)
    ]
}

gate(
    "C exact bilinear rank profile 8,0,0,8",
    block_ranks == {
        "WW": 8,
        "WU": 0,
        "UW": 0,
        "UU": 8
    }
)

gate(
    "C tensor kernel dimensions 56,64,64,56",
    {
        k: 64-v
        for k, v in block_ranks.items()
    }
    ==
    {
        "WW": 56,
        "WU": 64,
        "UW": 64,
        "UU": 56
    }
)


# ------------------------------------------------------------
# C2 — Verify Cayley-Dickson sector formula
# ------------------------------------------------------------

# For x=(a,b), y=(c,d):
#
#        P(xy) = ac - conjugate(d)*b
#
# The mixed input-sector contributions vanish.

def octmul(a, b):

    out = np.zeros(8, dtype=np.int64)

    for i in np.flatnonzero(a):
        for j in np.flatnonzero(b):

            out[int(IX[i, j])] += (
                int(SG[i, j])
                * int(a[i])
                * int(b[j])
            )

    return out


def conj8(a):

    c = a.copy()
    c[1:] *= -1
    return c


for i, j in it.product(range(8), repeat=2):

    x = np.zeros(16, dtype=np.int64)
    y = np.zeros(16, dtype=np.int64)

    x[8+i] = 1
    y[8+j] = 1

    ei = I8[:, i]
    ej = I8[:, j]

    if not np.array_equal(
        mul(x, y)[:8],
        -octmul(conj8(ej), ei)
    ):
        raise AssertionError(
            "FAIL | C UU bilinear identity"
        )

gate(
    "C exact UU = -conj(d)b on all 64 input basis pairs",
    True
)


# ------------------------------------------------------------
# C3 — Exact external-input projection defect
# ------------------------------------------------------------

# Delta_P(x,y) =
#     P(xy) - P((Px)(Py))
#
# For this splitting:
#
# Delta_P(x,y) = P((Kx)(Ky)).
#
# Verify on all 256 input-basis pairs.

for i, j in it.product(range(16), repeat=2):

    x = I16[:, i]
    y = I16[:, j]

    defect = (
        P @ mul(x, y)
        - P @ mul(P @ x, P @ y)
    )

    expected_defect = P @ mul(K @ x, K @ y)

    if not np.array_equal(
        defect,
        expected_defect
    ):
        raise AssertionError(
            "FAIL | C defect decomposition"
        )

gate(
    "C external-input projection defect exactly equals UU contribution",
    True
)

print("C | bilinear block ranks:", block_ranks)


# ============================================================
# GATE D — ASSOCIATOR PROJECTORS AND COMPOSITIONS
# ============================================================

print("\n--- GATE D: ASSOCIATOR PROJECTORS AND COMPOSITIONS ---")

L8 = [
    J[i][:8, :8]
    for i in range(8)
]

# Enumerate the quaternionic coordinate lines
# directly from the chosen multiplication index law.
# No frozen SIM14 Fano sector list imported.

quats = []

for a, b in it.combinations(range(1, 8), 2):

    c = a ^ b
    line = tuple(sorted((0, a, b, c)))

    if line not in quats:
        quats.append(line)

quats = sorted(quats)

gate(
    "D seven independently enumerated quaternionic coordinate sectors",
    len(quats) == 7
)

Rrecords = []

for line in quats:

    a, b = line[1], line[2]
    c = a ^ b

    # Associator defect:
    #
    # D_ab = L_(e_a e_b) - L_a L_b

    D = (
        int(SG[a, b]) * L8[c]
        - L8[a] @ L8[b]
    )

    Q = np.zeros((8, 8), dtype=np.int64)

    for i in line:
        Q[i, i] = 1

    # For normalized coordinate imaginary basis:
    #
    #     D^T D = 4(I-Q)

    if not np.array_equal(
        D.T @ D,
        4 * (I8-Q)
    ):
        raise AssertionError(
            "FAIL | D associator spectral projector"
        )

    # R = inclusion * Q * P
    #
    # This is a 16D projector whose image is a
    # 4D quaternionic coordinate subspace.

    R = np.zeros((16, 16), dtype=np.int64)
    R[:8, :8] = Q

    if (
        not np.array_equal(R @ R, R)
        or int(np.trace(R)) != 4
    ):
        raise AssertionError(
            "FAIL | D composed projection"
        )

    # Psi(x,y) = R(xy).
    # Compute its output ranks by input sector.

    ranks = {
        name: block_rank(rows, cols, set(line))
        for name, rows, cols in [
            ("WW", W, W),
            ("WU", W, U),
            ("UW", U, W),
            ("UU", U, U)
        ]
    }

    if ranks != {
        "WW": 4,
        "WU": 0,
        "UW": 0,
        "UU": 4
    }:
        raise AssertionError(
            "FAIL | D four-plane bilinear blocks"
        )

    # Inherited symplectic rank check.
    #
    # These restricted matrices are signed partial
    # permutations, so exact rank equals the number
    # of nonzero rows. No floating-point rank used.

    for i in range(1, 8):

        om = (-J[i])[np.ix_(line, line)]

        assert all(
            np.count_nonzero(row) <= 1
            for row in om
        )

        assert all(
            np.count_nonzero(col) <= 1
            for col in om.T
        )

        exact_rank = int(
            np.count_nonzero(
                np.any(om != 0, axis=1)
            )
        )

        if exact_rank != (4 if i in line else 0):
            raise AssertionError(
                "FAIL | D symplectic ranks"
            )

    Rrecords.append({
        "line": list(line),
        "rank_R": 4,
        "bilinear_blocks": ranks
    })


gate(
    "D all 7 associator projectors and composed Psi maps certified",
    True
)

gate(
    "D all 7 four-planes have exact inherited symplectic rank signature",
    True
)


# ============================================================
# GATE E — INDEPENDENT ARITHMETIC CONTROLS
# ============================================================

print("\n--- GATE E: INDEPENDENT ARITHMETIC CONTROLS ---")

# The following is a declared octavian-order input,
# NOT an intrinsically derived lattice.
#
# It reproduces the multiplication-compatible support
# arrangement of SIM17.1 in canonical CD coordinates.
#
# The index map is an input conversion ONLY.
# It is not used for frozen OPS graph selection.

SIM17_COORD = {
    0: 0,
    1: 4,
    2: 1,
    3: 5,
    4: 7,
    5: 3,
    6: 6,
    7: 2
}

SUPPORTS = (
    (0, 1, 2, 3),
    (0, 1, 4, 5),
    (0, 1, 6, 7),
    (0, 2, 4, 7),
    (0, 2, 5, 6),
    (0, 3, 4, 6),
    (0, 3, 5, 7),
    (1, 2, 4, 6),
    (1, 2, 5, 7),
    (1, 3, 4, 7),
    (1, 3, 5, 6),
    (2, 3, 4, 5),
    (2, 3, 6, 7),
    (4, 5, 6, 7)
)

CD_SUPPORTS = tuple(
    tuple(sorted(SIM17_COORD[i] for i in supp))
    for supp in SUPPORTS
)

code = (
    {0, 255}
    | {
        sum(1 << i for i in supp)
        for supp in CD_SUPPORTS
    }
)

gate(
    "E octavian 16-word doubly even additive support code",
    len(code) == 16
    and all(
        a ^ b in code
        for a, b in it.product(code, repeat=2)
    )
    and Counter(x.bit_count() for x in code)
    == {0: 1, 4: 14, 8: 1}
)


# ------------------------------------------------------------
# E2 — Generate the 240 norm-one vectors
# ------------------------------------------------------------

# Doubled octonion coordinates are used:
#
#   coordinate array represents 2*x.
#
# The 16 basis roots are +/-2e_i.
# The 224 half-unit roots have four +/-1 entries.

roots = set()

for i in range(8):
    for s in (-2, 2):

        u = [0]*8
        u[i] = s
        roots.add(tuple(u))

for supp in CD_SUPPORTS:

    for signs in it.product((-1, 1), repeat=4):

        u = [0]*8

        for i, s in zip(supp, signs):
            u[i] = s

        roots.add(tuple(u))

roots = sorted(roots)

gate(
    "E 240 norm-one doubled-coordinate octavian generators",
    len(roots) == 240
    and all(
        sum(x*x for x in root) == 4
        for root in roots
    )
)


# ------------------------------------------------------------
# E3 — Exact lattice membership
# ------------------------------------------------------------

def in_order2(a):
    """Test 2*x in the chosen octavian overlattice."""
    odd = sum(
        1 << i
        for i, x in enumerate(a)
        if int(x) % 2
    )
    return odd in code


def in_z8_2(a):
    """Test 2*x in the standard integer lattice."""
    return all(int(x) % 2 == 0 for x in a)


def octmul_raw2(a, b):
    """
    a,b represent doubled inputs 2*x, 2*y.
    Output represents 4*(xy).
    """
    out = [0]*8

    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    out[int(IX[i, j])] += (
                        int(ai)
                        * int(bj)
                        * int(SG[i, j])
                    )

    return out


def conj2(a):
    return (a[0],) + tuple(-v for v in a[1:])


unit_products = []

for a in roots:

    for b in roots:

        raw = octmul_raw2(a, b)

        if any(x % 2 for x in raw):
            raise AssertionError(
                "FAIL | E half-integrality of products"
            )

        out = tuple(x // 2 for x in raw)

        if not in_order2(out):
            raise AssertionError(
                "FAIL | E chosen order multiplication closure"
            )

        unit_products.append(out)

gate(
    "E 57600 octavian unit products lie in chosen order",
    len(unit_products) == 57600
)


# ------------------------------------------------------------
# E4 — Independently specified input lattice pairs
# ------------------------------------------------------------

# Compare:
#
# A = Z8, B = Z8
# A = O,  B = Z8
# A = Z8, B = O
# A = O,  B = O
#
# Test:
#
#       -conjugate(B)*B subset A
#
# On declared additive generator sets.
#
# This is an arithmetic admissibility comparison,
# not intrinsic selection of A or B.

ZGEN = [
    tuple(
        2 if k == i else 0
        for k in range(8)
    )
    for i in range(8)
]

O_GEN = (
    ZGEN
    + [
        tuple(
            1 if k in supp else 0
            for k in range(8)
        )
        for supp in CD_SUPPORTS
    ]
)


def cross_ok(A, B):

    for b, d in it.product(B, repeat=2):

        raw = octmul_raw2(conj2(d), b)

        if any(x % 2 for x in raw):
            return False

        out = tuple(x // 2 for x in raw)

        if not A(out):
            return False

    return True


arith = {
    "A=Z8_B=Z8": cross_ok(in_z8_2, ZGEN),
    "A=O_B=Z8": cross_ok(in_order2, ZGEN),
    "A=Z8_B=O": cross_ok(in_z8_2, O_GEN),
    "A=O_B=O": cross_ok(in_order2, O_GEN)
}

gate(
    "E independent lattice controls separate admissible/inadmissible pairs",
    arith == {
        "A=Z8_B=Z8": True,
        "A=O_B=Z8": True,
        "A=Z8_B=O": False,
        "A=O_B=O": True
    }
)

print("E | arithmetic controls:", arith)


# ============================================================
# GATE F — UNSIGNED DIRECTION/HALF-TURN SELECTION
# ============================================================

print("\n--- GATE F: UNSIGNED DIRECTION/HALF-TURN SELECTION ---")

# A signed octonionic multiplication generates an
# unsigned XOR action on the eight basis indices.
#
# Every nonzero direction defines one fixed-point-free
# involution. Do not preselect h or use C8.

translations = {
    a: tuple(x ^ a for x in range(8))
    for a in range(1, 8)
}

gate(
    "F seven distinct fixed-point-free involutions",
    len(set(translations.values())) == 7
    and all(
        all(
            t[t[x]] == x and t[x] != x
            for x in range(8)
        )
        for t in translations.values()
    )
)

# The parent symmetry group certified in Gate B
# acts transitively on all seven directions.
#
# In particular, preserving P and Phi alone does
# not permit a unique invariant direction.

permutation_images = {
    a: {
        perm[a]
        for perm, _ in autos
    }
    for a in range(1, 8)
}

gate(
    "F no invariant unique translation under residual automorphisms",
    all(
        ims == set(range(1, 8))
        for ims in permutation_images.values()
    )
)

print("F | all seven candidate translations remain equivalent")


# ============================================================
# GATE G — HELD-OUT OPS ACCEPTANCE
# ============================================================

print("\n--- GATE G: HELD-OUT OPS ACCEPTANCE ---")

# Frozen target enters here for the first time.
# None of these objects participate in Gates A-F.

FROZEN_EDGES = {
    (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 7)
}

FROZEN_P = (
    1, 0, 3, 2, 5, 4, 7, 6
)

FROZEN_H = (
    7, 6, 5, 4, 3, 2, 1, 0
)

FROZEN_OMEGA_PAIRS = (
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7)
)

# For comparison only, determine which canonical
# XOR translations happen to match the frozen
# h permutation under raw integer labeling.
#
# Matching as a member of a seven-element family
# does not constitute independent selection.

raw_h_matches = [
    a
    for a, t in translations.items()
    if t == FROZEN_H
]

print(
    "G | frozen h matches in raw canonical XOR indexing:",
    raw_h_matches
)

print(
    "G | selected h: NO; graph and signed Omega: NOT DERIVED"
)


# ============================================================
# MACHINE-READABLE TRUTH PACKET
# ============================================================

packet = {
    "sim": "17.4",

    "status": "EXACT_RESTRICTED_AUDIT",

    "gamma7_minus_gamma8": True,

    "projector_rank": 8,

    "parent_automorphism_subgroup_size": len(autos),

    "full_parent_automorphism_group_classified": False,

    "closed_coordinate_carrier_count": len(closed),

    "ambient_full_clifford_carrier_count":
        ambient.count(21),

    "parent_projector_intrinsically_selected": False,

    "bilinear_blocks": block_ranks,

    "bilinear_block_tensor_kernels": {
        k: 64-v
        for k, v in block_ranks.items()
    },

    "bilinear_defect_equals_UU": True,

    "quaternionic_4planes": Rrecords,

    "lattice_controls": arith,

    "octavian_supports_supplied": True,

    "octavian_unit_products_tested":
        len(unit_products),

    "eight_state_quotient_supplied_by_coordinate_basis":
        True,

    "translation_candidates": len(translations),

    "translation_orbit_under_certified_subgroup": 7,

    "h_intrinsically_selected": False,

    "frozen_h_raw_index_coincidences": raw_h_matches,

    "exact_frozen_C8_selected": False,

    "p_h_signed_Omega_recovered": False,

    "full_OPS_independently_derived": False,

    "input_provenance": {
        "parent":
            "Fixed standard Cayley-Dickson multiplication and Euclidean metric",

        "projector":
            "Seven supplied generator family; alternative eight-family gives same eigenspaces",

        "arithmetic":
            "SIM17.1 chosen octavian support code; Z8 controls independently frozen",

        "vertex_labels":
            "Standard Cayley-Dickson coordinate basis, NOT SIM14 labels",

        "symmetry":
            "1344 signed-octonion automorphisms certified as 16D lifts; NOT full parent group",

        "F_gate":
            "Seven unmarked quotient XOR translations; no graph or h",

        "G_gate":
            "Frozen OPS first introduced as held-out target",

        "dynamics":
            "NONE; exact algebraic admissibility only"
    }
}


try:
    with open(
        "SIM17.4_truth_packet.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(packet, f, indent=2)

    print(
        "Truth packet saved: SIM17.4_truth_packet.json"
    )

except OSError as exc:
    print("Truth packet could not be written:", exc)


print(
    "TRUTH_PACKET_SUMMARY="
    + json.dumps(packet, sort_keys=True)
)


# ============================================================
# FINAL VALIDATOR VERDICT
# ============================================================

print("\n" + "=" * 64)
print("SIM17.4 — FINAL VALIDATOR VERDICT")
print("=" * 64)

print("Clifford volume identities:      PASS")
print("Parent spectral projectors:      PASS")
print("Bilinear 8D sector blocks:       PASS")
print("Composed 4D projectors:          PASS")
print("Octavian arithmetic controls:    PASS")
print("Half-turn intrinsic selection:   NOT ESTABLISHED")
print("Exact frozen OPS recovery:       NOT ESTABLISHED")

print("\nEND SIM17.4")







"""




RESULTS:





================================================================
SIM17.4 — PARENT PROJECTION / BILINEAR / HALF-TURN AUDIT
================================================================

--- GATE A: PARENT AND CLIFFORD VOLUME ---PASS | A signed multiplication and 16D identity
PASS | A individual 15 imaginary orthogonal complex structures
PASS | A full 7-generator parent anticommutation
PASS | A Gamma7 = diag(-I8,+I8)
PASS | A Gamma8 = -Gamma7
PASS | A exact complementary orthogonal spectral projectors
A | rank P = 8, rank K = 8

--- GATE B: PROJECTOR SELECTION AND INPUT SYMMETRY ---PASS | B 15 closed coordinate 8-planes
PASS | B only standard W has 21 ambient anticommuting pairs
PASS | B 1344 signed octonion basis automorphisms
PASS | B all 1344 octonion automorphisms lift and fix P
PASS | B residual parent symmetry transitive on seven directions
B | certified parent automorphism subgroup: 1344

--- GATE C: FOUR BILINEAR SECTOR BLOCKS ---PASS | C exact bilinear rank profile 8,0,0,8
PASS | C tensor kernel dimensions 56,64,64,56
PASS | C exact UU = -conj(d)b on all 64 input basis pairs
PASS | C external-input projection defect exactly equals UU contribution
C | bilinear block ranks: {'WW': 8, 'WU': 0, 'UW': 0, 'UU': 8}

--- GATE D: ASSOCIATOR PROJECTORS AND COMPOSITIONS ---PASS | D seven independently enumerated quaternionic coordinate sectors
PASS | D all 7 associator projectors and composed Psi maps certified
PASS | D all 7 four-planes have exact inherited symplectic rank signature

--- GATE E: INDEPENDENT ARITHMETIC CONTROLS ---PASS | E octavian 16-word doubly even additive support code
PASS | E 240 norm-one doubled-coordinate octavian generators
PASS | E 57600 octavian unit products lie in chosen order
PASS | E independent lattice controls separate admissible/inadmissible pairs
E | arithmetic controls: {'A=Z8_B=Z8': True, 'A=O_B=Z8': True, 'A=Z8_B=O': False, 'A=O_B=O': True}

--- GATE F: UNSIGNED DIRECTION/HALF-TURN SELECTION ---PASS | F seven distinct fixed-point-free involutions
PASS | F no invariant unique translation under residual automorphisms
F | all seven candidate translations remain equivalent

--- GATE G: HELD-OUT OPS ACCEPTANCE ---G | frozen h matches in raw canonical XOR indexing: [7]
G | selected h: NO; graph and signed Omega: NOT DERIVED
Truth packet saved: SIM17.4_truth_packet.json
TRUTH_PACKET_SUMMARY={"ambient_full_clifford_carrier_count": 1, "bilinear_block_tensor_kernels": {"UU": 56, "UW": 64, "WU": 64, "WW": 56}, "bilinear_blocks": {"UU": 8, "UW": 0, "WU": 0, "WW": 8}, "bilinear_defect_equals_UU": true, "closed_coordinate_carrier_count": 15, "eight_state_quotient_supplied_by_coordinate_basis": true, "exact_frozen_C8_selected": false, "frozen_h_raw_index_coincidences": [7], "full_OPS_independently_derived": false, "full_parent_automorphism_group_classified": false, "gamma7_minus_gamma8": true, "h_intrinsically_selected": false, "input_provenance": {"F_gate": "Seven unmarked quotient XOR translations; no graph or h", "G_gate": "Frozen OPS first introduced as held-out target", "arithmetic": "SIM17.1 chosen octavian support code; Z8 controls independently frozen", "dynamics": "NONE; exact algebraic admissibility only", "parent": "Fixed standard Cayley-Dickson multiplication and Euclidean metric", "projector": "Seven supplied generator family; alternative eight-family gives same eigenspaces", "symmetry": "1344 signed-octonion automorphisms certified as 16D lifts; NOT full parent group", "vertex_labels": "Standard Cayley-Dickson coordinate basis, NOT SIM14 labels"}, "lattice_controls": {"A=O_B=O": true, "A=O_B=Z8": true, "A=Z8_B=O": false, "A=Z8_B=Z8": true}, "octavian_supports_supplied": true, "octavian_unit_products_tested": 57600, "p_h_signed_Omega_recovered": false, "parent_automorphism_subgroup_size": 1344, "parent_projector_intrinsically_selected": false, "projector_rank": 8, "quaternionic_4planes": [{"bilinear_blocks": {"UU": 4, "UW": 0, "WU": 0, "WW": 4}, "line": [0, 1, 2, 3], "rank_R": 4}, {"bilinear_blocks": {"UU": 4, "UW": 0, "WU": 0, "WW": 4}, "line": [0, 1, 4, 5], "rank_R": 4}, {"bilinear_blocks": {"UU": 4, "UW": 0, "WU": 0, "WW": 4}, "line": [0, 1, 6, 7], "rank_R": 4}, {"bilinear_blocks": {"UU": 4, "UW": 0, "WU": 0, "WW": 4}, "line": [0, 2, 4, 6], "rank_R": 4}, {"bilinear_blocks": {"UU": 4, "UW": 0, "WU": 0, "WW": 4}, "line": [0, 2, 5, 7], "rank_R": 4}, {"bilinear_blocks": {"UU": 4, "UW": 0, "WU": 0, "WW": 4}, "line": [0, 3, 4, 7], "rank_R": 4}, {"bilinear_blocks": {"UU": 4, "UW": 0, "WU": 0, "WW": 4}, "line": [0, 3, 5, 6], "rank_R": 4}], "sim": "17.4", "status": "EXACT_RESTRICTED_AUDIT", "translation_candidates": 7, "translation_orbit_under_certified_subgroup": 7}

================================================================SIM17.4 — FINAL VALIDATOR VERDICT
================================================================
Clifford volume identities:      PASS
Parent spectral projectors:      PASS
Bilinear 8D sector blocks:       PASS
Composed 4D projectors:          PASS
Octavian arithmetic controls:    PASS
Half-turn intrinsic selection:   NOT ESTABLISHED
Exact frozen OPS recovery:       NOT ESTABLISHED

END SIM17.4
