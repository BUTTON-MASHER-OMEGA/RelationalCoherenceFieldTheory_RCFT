
#!/usr/bin/env python3
"""
SIM17.3
Bilinear Constraints, Symplectic Inheritance,
and Cycle-Cocycle Audit

Exact finite algebra and linear algebra.

Scope:
    16D Cayley-Dickson algebra
    15 multiplication-derived complex structures
    15 inherited symplectic forms
    Ambient versus restricted Clifford compatibility
    Unital coordinate eight-plane census
    Quaternionic four-plane incidence
    Signed multiplication cochains
    Four-cycle negative controls
    Parent symmetries and projection admissibility
    Held-out frozen OPS acceptance

No dynamics, ISP, stochastic evolution, graph fitting,
or renormalization.

Requires: numpy
"""

import itertools as it
import json
from collections import Counter
from math import comb

import numpy as np


# ============================================================
# RUN-BUTTON COMPATIBILITY
# ============================================================

def gate(name, condition):
    if not bool(condition):
        raise AssertionError("FAIL | " + name)
    print("PASS | " + name)


print("=" * 65)
print("SIM17.3 — BILINEAR CONSTRAINTS AND SYMPLECTIC INHERITANCE")
print("=" * 65)


# ============================================================
# GATE A — EXACT 16D CAYLEY-DICKSON ALGEBRA
# ============================================================

def cayley_dickson(i, j, n=16):
    if n == 1:
        return (1, 0)

    k = n // 2

    if i < k and j < k:
        return cayley_dickson(i, j, k)

    if i < k and j >= k:
        sg, t = cayley_dickson(j-k, i, k)
        return sg, t+k

    if i >= k and j < k:
        sg, t = cayley_dickson(i-k, j, k)
        return sg * (1 if j == 0 else -1), t+k

    sg, t = cayley_dickson(j-k, i-k, k)
    return -sg * (1 if j-k == 0 else -1), t


SG = np.zeros((16, 16), dtype=np.int64)
IX = np.zeros((16, 16), dtype=np.int64)

for i in range(16):
    for j in range(16):
        SG[i, j], IX[i, j] = cayley_dickson(i, j)

I16 = np.eye(16, dtype=np.int64)


def associator(i, j, k):
    """
    (e_i e_j)e_k - e_i(e_j e_k)
    """
    q = {}

    ab = int(IX[i, j])
    bc = int(IX[j, k])

    v = int(IX[ab, k])
    w = int(IX[i, bc])

    q[v] = (
        q.get(v, 0)
        + int(SG[i, j]) * int(SG[ab, k])
    )

    q[w] = (
        q.get(w, 0)
        - int(SG[j, k]) * int(SG[i, bc])
    )

    return {
        a: b for a, b in q.items() if b != 0
    }


def left(i):
    M = np.zeros((16, 16), dtype=np.int64)

    for j in range(16):
        M[int(IX[i, j]), j] = SG[i, j]

    return M


J = [left(i) for i in range(16)]


def signed_mul(x, y):
    out = np.zeros(16, dtype=np.int64)

    for i in np.flatnonzero(x):
        for j in np.flatnonzero(y):
            out[int(IX[i, j])] += (
                int(x[i])
                * int(y[j])
                * int(SG[i, j])
            )

    return out


print("\n--- GATE A: CAYLEY-DICKSON MULTIPLICATION ---")

gate(
    "A signed multiplication index law",
    all(
        int(IX[i, j]) == (i ^ j)
        for i in range(16)
        for j in range(16)
    )
)

gate(
    "A identity and imaginary squares",
    all(
        int(SG[0, i]) == int(SG[i, 0]) == 1
        for i in range(16)
    )
    and all(
        int(SG[i, i]) == -1
        and int(IX[i, i]) == 0
        for i in range(1, 16)
    )
)

gate(
    "A explicit nonassociativity",
    bool(associator(1, 2, 9))
)

gate(
    "A exact bilinear structure constants",
    all(abs(int(s)) == 1 for s in SG.flat)
)


# ============================================================
# GATE B — COMPLEX AND SYMPLECTIC STRUCTURES
# ============================================================

print("\n--- GATE B: 16D COMPLEX/SYMPLECTIC OPERATORS ---")

individual = []

for i in range(1, 16):
    L = J[i]

    individual.append(
        bool(
            np.array_equal(L @ L, -I16)
            and np.array_equal(L.T, -L)
            and np.array_equal(L.T @ L, I16)
        )
    )

gate(
    "B all 15 imaginary left operators square to -I",
    len(individual) == 15 and all(individual)
)

forms = [-J[i] for i in range(1, 16)]

gate(
    "B all 15 forms are symplectic and metric-compatible",
    all(
        np.array_equal(F.T, -F)
        and np.array_equal(F @ F, -I16)
        and np.array_equal(F @ J[i], I16)
        for i, F in enumerate(forms, 1)
    )
)


# ------------------------------------------------------------
# B2 — Exact octonionic block decomposition
# ------------------------------------------------------------

C = np.diag([1] + [-1]*7).astype(np.int64)

L8 = [
    J[a][:8, :8]
    for a in range(8)
]

R8 = []

for a in range(8):
    R = np.zeros((8, 8), dtype=np.int64)

    for x in range(8):
        R[int(IX[x, a]), x] = SG[x, a]

    R8.append(R)


gate(
    "B original octonion block formula",
    all(
        np.array_equal(
            J[a],
            np.block([
                [
                    L8[a],
                    np.zeros((8, 8), dtype=np.int64)
                ],
                [
                    np.zeros((8, 8), dtype=np.int64),
                    R8[a]
                ]
            ])
        )
        for a in range(1, 8)
    )
)

gate(
    "B doubled octonion block formula",
    all(
        np.array_equal(
            J[8+b],
            np.block([
                [
                    np.zeros((8, 8), dtype=np.int64),
                    -R8[b] @ C
                ],
                [
                    L8[b] @ C,
                    np.zeros((8, 8), dtype=np.int64)
                ]
            ])
        )
        for b in range(8)
    )
)


# ------------------------------------------------------------
# B3 — Full 105-pair Clifford compatibility census
# ------------------------------------------------------------

pairs = []
incompatible = []
partition = Counter()

for i, j in it.combinations(range(1, 16), 2):

    anti = J[i] @ J[j] + J[j] @ J[i]

    family = (
        "AA" if i < 8 and j < 8
        else "BB" if i >= 8
        else "AB"
    )

    if not np.any(anti):
        pairs.append((i, j))
        partition[family] += 1

    else:
        rr, cc = np.argwhere(anti != 0)[0]

        incompatible.append({
            "pair": [i, j],
            "witness_input": int(cc),
            "witness_output": int(rr),
            "coefficient": int(anti[rr, cc])
        })


gate(
    "B exactly 63 of 105 pairs anticommute",
    len(pairs) == 63
    and len(incompatible) == 42
    and dict(partition) == {
        "AA": 21,
        "BB": 28,
        "AB": 14
    }
)

witness = (
    J[1] @ J[10] + J[10] @ J[1]
)[:, 4]

gate(
    "B exact nonanticommutation witness",
    np.count_nonzero(witness) == 1
    and int(witness[15]) == -2
)

print("B | pair partition:", dict(partition))
print("B | nonanticommuting pairs:", len(incompatible))


# ============================================================
# GATE C — PARENT-LEVEL CLIFFORD COMPATIBILITY
# ============================================================

print("\n--- GATE C: COORDINATE CARRIER CENSUS ---")

closed = []
CROWS = []
QUATS = {}

for comb7 in it.combinations(range(1, 16), 7):

    W = (0,) + comb7
    WS = set(W)

    is_closed = all(
        int(IX[i, j]) in WS
        for i in W
        for j in W
    )

    if not is_closed:
        continue

    closed.append(W)

    # Alternativity test on the full coordinate basis.
    # Trilinearity extends these identities to the carrier.

    alt = all(
        associator(a, a, b) == {}
        and associator(b, a, a) == {}
        and associator(a, b, c) == {
            k: -v
            for k, v in associator(b, a, c).items()
        }
        for a in W
        for b in W
        for c in W
    )

    ambient = 0
    restricted = 0

    for i, j in it.combinations(W[1:], 2):

        ant = J[i] @ J[j] + J[j] @ J[i]

        if not np.any(ant):
            ambient += 1

        if not np.any(ant[np.ix_(W, W)]):
            restricted += 1


    # Identify associative quaternionic coordinate sectors
    # using multiplication and associator tests.

    quats = []

    for triple in it.combinations(W[1:], 3):

        Q = (0,) + triple

        if (
            all(
                int(IX[i, j]) in Q
                for i in Q
                for j in Q
            )
            and all(
                associator(i, j, k) == {}
                for i in Q
                for j in Q
                for k in Q
            )
        ):
            quats.append(Q)

    QUATS[W] = quats


    # Separate the three maps:
    #
    # C_W     = (I - pi_W)(xy), x,y in W
    # Delta_W = pi_W(xy) - pi_W(pi_W x pi_W y)
    # Phi_W   = pi_W(xy)
    #
    # Rank means rank on the tensor-product domain.

    c_outputs = {
        int(IX[i, j])
        for i in W
        for j in W
        if int(IX[i, j]) not in WS
    }

    delta_outputs = {
        int(IX[i, j])
        for i in range(16)
        for j in range(16)
        if int(IX[i, j]) in WS
        and not (i in WS and j in WS)
    }

    phi_outputs = {
        int(IX[i, j])
        for i in range(16)
        for j in range(16)
        if int(IX[i, j]) in WS
    }

    CROWS.append({
        "basis": list(W),
        "alternative": bool(alt),
        "closure_defect_rank": len(c_outputs),
        "projection_defect_rank": len(delta_outputs),
        "bilinear_phi_rank": len(phi_outputs),
        "ambient_clifford_pairs": ambient,
        "restricted_clifford_pairs": restricted,
        "quaternionic_coordinate_4planes": [
            list(q) for q in quats
        ]
    })


gate(
    "C all 6435 coordinate candidates considered",
    comb(15, 7) == 6435
    and len(closed) == 15
)

class_counts = Counter(
    (
        r["alternative"],
        r["ambient_clifford_pairs"],
        r["restricted_clifford_pairs"]
    )
    for r in CROWS
)

expected = {
    (True, 21, 21): 1,
    (True, 15, 21): 7,
    (False, 9, 9): 7
}

gate(
    "C exact 1+7+7 Clifford compatibility classes",
    dict(class_counts) == expected
)

selected = [
    r for r in CROWS
    if r["ambient_clifford_pairs"] == 21
]

gate(
    "C unique coordinate carrier with full ambient compatibility",
    len(selected) == 1
    and selected[0]["basis"] == list(range(8))
)

gate(
    "C closure/projection/bilinear ranks separated",
    all(
        (
            r["closure_defect_rank"],
            r["projection_defect_rank"],
            r["bilinear_phi_rank"]
        ) == (0, 8, 8)
        for r in CROWS
    )
)

gate(
    "C seven quaternionic sectors per alternative carrier",
    all(
        len(QUATS[tuple(r["basis"])]) == 7
        for r in CROWS
        if r["alternative"]
    )
)

print(
    "C | compatibility classes:",
    {str(k): v for k, v in class_counts.items()}
)


# ============================================================
# GATE D — SYMPLECTIC INHERITANCE
# ============================================================

print("\n--- GATE D: 16D TO 8D TO 4D INHERITANCE ---")


def skew_rank_integer(M):
    """
    Exact rank for signed partial-permutation matrices.

    Every restricted multiplication-derived form has
    at most one nonzero element per row and column.

    Its rank is exactly its number of occupied rows.
    """

    rows = np.flatnonzero(
        np.any(M != 0, axis=1)
    )

    cols = np.flatnonzero(
        np.any(M != 0, axis=0)
    )

    assert all(
        np.count_nonzero(M[k, :]) <= 1
        for k in range(M.shape[0])
    )

    assert all(
        np.count_nonzero(M[:, k]) <= 1
        for k in range(M.shape[1])
    )

    assert len(rows) == len(cols)

    return int(len(rows))


DROWS = []

for W in closed:

    eight = [
        skew_rank_integer(F[np.ix_(W, W)])
        for F in forms
    ]

    assert all(
        eight[i-1] == (8 if i in W else 0)
        for i in range(1, 16)
    )

    qrecords = []

    for Q in QUATS[W]:

        ranks = [
            skew_rank_integer(F[np.ix_(Q, Q)])
            for F in forms
        ]

        assert all(
            ranks[i-1] == (4 if i in Q else 0)
            for i in range(1, 16)
        )

        qrecords.append({
            "indices": list(Q),
            "ranks": ranks
        })

    DROWS.append({
        "carrier": list(W),
        "omega_ranks": eight,
        "omega_kernel_dimensions": [
            8-x for x in eight
        ],
        "quaternionic_ranks": qrecords
    })


gate(
    "D all 15 inherited forms classified on every 8-plane",
    len(DROWS) == 15
    and all(
        Counter(row["omega_ranks"]) == {
            0: 8,
            8: 7
        }
        for row in DROWS
    )
)

gate(
    "D quaternionic four-plane Fano incidence ranks",
    all(
        Counter(q["ranks"]) == {
            0: 12,
            4: 3
        }
        for row in DROWS
        for q in row["quaternionic_ranks"]
    )
)

gate(
    "D canonical carrier contains seven associative four-planes",
    len(QUATS[tuple(range(8))]) == 7
)


# Finite and infinitesimal symplectic checks are distinct.

gate(
    "D finite symplectic transformations and Lie generators",
    all(
        np.array_equal(
            J[i].T @ forms[i-1] @ J[i],
            forms[i-1]
        )
        and np.array_equal(
            J[i].T @ forms[i-1]
            + forms[i-1] @ J[i],
            np.zeros((16, 16), dtype=np.int64)
        )
        for i in range(1, 16)
    )
)

print("D | rank-8 inherited forms per carrier: 7")
print("D | rank-4 inherited forms per quaternionic sector: 3")


# ============================================================
# GATE E — COCYCLES AND FOUR-CYCLE DIAGNOSTICS
# ============================================================

print("\n--- GATE E: COCHAIN AND CYCLE AUDIT ---")


def F(a, b):
    return int(SG[a, b])


def alpha(a, b, c):
    """
    Signed associativity defect:

    (e_a e_b)e_c / e_a(e_b e_c).

    F is a 2-cochain, not generally a 2-cocycle.

    alpha is its multiplicative 3-coboundary.
    """

    return (
        F(a, b)
        * F(a ^ b, c)
        * F(b, c)
        * F(a, b ^ c)
    )


nontrivial = [
    (a, b, c)
    for a in range(16)
    for b in range(16)
    for c in range(16)
    if alpha(a, b, c) == -1
]

gate(
    "E nontrivial associativity 3-coboundary",
    bool(nontrivial)
)

gate(
    "E normalized 3-coboundary",
    all(
        alpha(0, a, b) == 1
        and alpha(a, 0, b) == 1
        and alpha(a, b, 0) == 1
        for a in range(16)
        for b in range(16)
    )
)


# Full 16^4 exact 3-cocycle identity.

gate(
    "E 3-cocycle identity on all 65536 quadruples",
    all(
        alpha(b, c, d)
        * alpha(a ^ b, c, d)
        * alpha(a, b ^ c, d)
        * alpha(a, b, c ^ d)
        * alpha(a, b, c)
        == 1
        for a in range(16)
        for b in range(16)
        for c in range(16)
        for d in range(16)
    )
)


q_count = 0

for W in closed:
    for Q in QUATS[W]:

        if all(
            alpha(a, b, c) == 1
            for a in Q
            for b in Q
            for c in Q
        ):
            q_count += 1
        else:
            raise AssertionError(
                "Quaternionic cocycle defect nonzero"
            )

gate(
    "E every quaternionic restriction has trivial coboundary",
    q_count == sum(
        len(x) for x in QUATS.values()
    )
)


# ------------------------------------------------------------
# E2 — Uniform unsigned translation controls
# ------------------------------------------------------------

def components(vertices, edges):

    adj = {
        v: set()
        for v in vertices
    }

    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    todo = set(vertices)
    sizes = []

    while todo:

        start = todo.pop()
        stack = [start]
        n = 0

        while stack:

            x = stack.pop()
            n += 1

            for y in adj[x]:
                if y in todo:
                    todo.remove(y)
                    stack.append(y)

        sizes.append(n)

    return sorted(sizes)


translation_trials = 0

for W in closed:

    for a, b in it.combinations(W[1:], 2):

        edges = {
            tuple(sorted((x, x ^ shift)))
            for x in W
            for shift in (a, b)
        }

        valid = (
            len(edges) == 8
            and components(W, edges) == [4, 4]
            and all(
                sum(x in edge for edge in edges) == 2
                for x in W
            )
        )

        if not valid:
            raise AssertionError(
                "Unexpected unsigned translation graph"
            )

        translation_trials += 1


gate(
    "E all 315 translation controls give two C4 components",
    translation_trials == 15 * 21
)


# ============================================================
# GATE F — PARENT NECESSITY AND NEGATIVE CONTROLS
# ============================================================

print("\n--- GATE F: PARENT NECESSITY ---")


# ------------------------------------------------------------
# F1 — Exact sedenion zero divisor
# ------------------------------------------------------------

zero_witness = None

for i, j in it.combinations(range(1, 16), 2):

    for k, l in it.combinations(range(1, 16), 2):

        if (i ^ k) != (j ^ l):
            continue

        if (i ^ l) != (j ^ k):
            continue

        for si, sj in it.product((-1, 1), repeat=2):

            x = np.zeros(16, dtype=np.int64)
            y = np.zeros(16, dtype=np.int64)

            x[i] = 1
            x[j] = si

            y[k] = 1
            y[l] = sj

            if not np.any(signed_mul(x, y)):

                zero_witness = {
                    "x": [[i, 1], [j, si]],
                    "y": [[k, 1], [l, sj]]
                }
                break

        if zero_witness:
            break

    if zero_witness:
        break


gate(
    "F nonzero parent zero-divisor witness",
    zero_witness is not None
)


# ------------------------------------------------------------
# F2 — Parent multiplication automorphism controls
# ------------------------------------------------------------

character_count = 0

for t in range(16):

    signs = [
        (-1)**((t & a).bit_count() % 2)
        for a in range(16)
    ]

    if all(
        signs[i] * signs[j] == signs[i ^ j]
        for i in range(16)
        for j in range(16)
    ):
        character_count += 1


gate(
    "F sixteen parent-algebra sign-character automorphisms",
    character_count == 16
)


# ------------------------------------------------------------
# F3 — Independent coordinate sign rephasing
# ------------------------------------------------------------

rephase = np.array(
    [1] + [
        -1 if i % 3 == 1 else 1
        for i in range(1, 16)
    ],
    dtype=np.int64
)

D = np.diag(rephase)

J_rephased = {
    i: int(rephase[i]) * (D @ J[i] @ D)
    for i in range(1, 16)
}

rephased_pairs = {
    (i, j)
    for i, j in it.combinations(range(1, 16), 2)
    if not np.any(
        J_rephased[i] @ J_rephased[j]
        + J_rephased[j] @ J_rephased[i]
    )
}

gate(
    "F sign-rephasing preserves exact Clifford compatibility",
    len(set(int(x) for x in rephase)) == 2
    and rephased_pairs == set(pairs)
)


# ------------------------------------------------------------
# F4 — Uniform algebraic adjacency rules
# ------------------------------------------------------------

# No frozen OPS information has been introduced.
#
# Rule 1: A pair is adjacent if it has a nonzero
#         associator witness on its carrier.
#
# Rule 2: A pair is adjacent if its restricted
#         left multiplication operators do not commute.
#
# These rules are exploratory and are not privileged
# physical selection laws.

RULES = (
    "associator_nonzero",
    "left_noncommuting"
)

adjacency_records = []

for W in closed:

    local = [
        J[i][np.ix_(W, W)]
        for i in W
    ]

    rec = {
        "carrier": list(W)
    }

    for rule in RULES:

        edges = []

        for u, v in it.combinations(range(8), 2):

            if rule == "associator_nonzero":

                event = any(
                    bool(associator(W[u], W[v], c))
                    for c in W
                )

            else:

                event = bool(
                    np.any(
                        local[u] @ local[v]
                        - local[v] @ local[u]
                    )
                )

            if event:
                edges.append((u, v))

        degrees = [
            sum(k in edge for edge in edges)
            for k in range(8)
        ]

        rec[rule] = {
            "edges": [list(e) for e in edges],
            "degrees": degrees,
            "components": components(range(8), edges)
        }

    adjacency_records.append(rec)


gate(
    "F graph inventory produced for all 15 carriers and both rules",
    len(adjacency_records) == 15
    and all(
        all(rule in row for rule in RULES)
        for row in adjacency_records
    )
)


# ============================================================
# GATE G — HELD-OUT OPS ACCEPTANCE
# ============================================================

print("\n--- GATE G: HELD-OUT OPS ACCEPTANCE ---")

# FROZEN TARGET INTRODUCED HERE ONLY.

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


def is_cycle8(edges):

    if len(edges) != 8:
        return False

    return (
        all(
            sum(v in edge for edge in edges) == 2
            for v in range(8)
        )
        and components(range(8), edges) == [8]
    )


cycle_inventory = []
exact = []

for row in adjacency_records:

    for rule in RULES:

        E = {
            tuple(pair)
            for pair in row[rule]["edges"]
        }

        if is_cycle8(E):

            cycle_inventory.append({
                "carrier": row["carrier"],
                "rule": rule,
                "edges": sorted(E)
            })

        if E == FROZEN_EDGES:

            exact.append({
                "carrier": row["carrier"],
                "rule": rule
            })


print(
    "G | cycle candidates:",
    len(cycle_inventory)
)

print(
    "G | exact labeled frozen C8:",
    len(exact)
)


# ============================================================
# PROVENANCE LEDGER
# ============================================================

ledger = {
    "parent": (
        "Fixed 16D Cayley-Dickson signed multiplication "
        "and Euclidean coordinate metric"
    ),

    "coordinates": (
        "Given parent signed basis; "
        "not independently derived"
    ),

    "carrier_search": (
        "All 6435 unital coordinate eight-planes; "
        "not all real eight-planes"
    ),

    "symplectic": (
        "Fifteen parent forms -L_e_i; "
        "Euclidean metric supplied"
    ),

    "quaternions": (
        "Derived from closure and associativity; "
        "no imported Fano sector list"
    ),

    "clifford": (
        "Ambient left-multiplication anticommutators; "
        "distinct from SIM17.2 Cl(8,0)"
    ),

    "adjacency": (
        "Two uniform exploratory extraction rules; "
        "neither privileged"
    ),

    "parent_symmetry": (
        "Sixteen diagonal-character automorphisms; "
        "full parent group not classified"
    ),

    "OPS": (
        "Frozen edges, p, h and signed Omega "
        "introduced only in Gate G"
    ),

    "selection": (
        "Complete ambient Clifford compatibility "
        "is an optional admissibility condition"
    ),

    "physics": (
        "No dynamical or physical mechanism claimed"
    )
}


# ============================================================
# MACHINE-READABLE TRUTH PACKET
# ============================================================

packet = {
    "sim": "17.3",

    "scope": "EXACT_RESTRICTED_COORDINATE_AUDIT",

    "individual_complex_structures": sum(individual),

    "parent_anticommuting_pairs": len(pairs),

    "parent_anticommuting_partition": dict(partition),

    "nonanticommuting_witnesses": incompatible,

    "closed_coordinate_carriers": len(closed),

    "carrier_clifford_classes": [
        {
            "alternative": a,
            "ambient": b,
            "restricted": c,
            "count": n
        }
        for (a, b, c), n in sorted(class_counts.items())
    ],

    "unique_full_ambient_clifford_coordinate_carrier":
        selected[0]["basis"],

    "canonical_selection_basis_dependent": True,

    "symplectic_forms_on_parent": 15,

    "symplectic_eight_plane_rank_pattern": {
        "rank8": 7,
        "rank0": 8
    },

    "symplectic_quaternionic_rank_pattern": {
        "rank4": 3,
        "rank0": 12
    },

    "cocycle_triples_tested": 16**3,

    "cocycle_quadruples_tested": 16**4,

    "nontrivial_3_coboundary_triples": len(nontrivial),

    "unsigned_translation_controls": translation_trials,

    "parent_zero_divisor_witness": zero_witness,

    "parent_sign_character_automorphisms":
        character_count,

    "cycle_candidates": cycle_inventory,

    "exact_frozen_graph_matches": exact,

    "full_OPS_derived": False,

    "carrier_intrinsically_selected": False,

    "projection_intrinsically_selected": False,

    "bilinear_law_intrinsically_selected": False,

    "distinguished_symplectic_form_selected": False,

    "vertex_set_intrinsically_selected": False,

    "adjacency_selected": False,

    "frozen_p_h_signed_Omega_recovered": False,

    "input_provenance": ledger,

    "carrier_inventory": CROWS,

    "symplectic_inventory": DROWS,

    "adjacency_inventory": adjacency_records
}


# Save if the Run environment permits file writes.

try:

    with open(
        "SIM17.3_truth_packet.json",
        "w",
        encoding="utf-8"
    ) as fp:

        json.dump(packet, fp, indent=2)

    print("Truth packet saved: SIM17.3_truth_packet.json")

except (OSError, PermissionError) as exc:

    print("Truth packet file unavailable:", exc)


# Always print a machine-readable summary.

excluded = {
    "carrier_inventory",
    "symplectic_inventory",
    "adjacency_inventory",
    "nonanticommuting_witnesses"
}

summary = {
    k: v
    for k, v in packet.items()
    if k not in excluded
}

print(
    "TRUTH_PACKET_SUMMARY="
    + json.dumps(summary, sort_keys=True)
)


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 65)
print("SIM17.3 — FINAL VALIDATOR VERDICT")
print("=" * 65)

print(
    "16D multiplication:                 PASS"
)

print(
    "Individual complex structures:      15/15 PASS"
)

print(
    "Multiplication-derived symplectic:  15/15 PASS"
)

print(
    "Joint anticommutation:              63/105"
)

print(
    "Closed coordinate carriers:        15"
)

print(
    "Full ambient Clifford carrier:     UNIQUE IN CENSUS"
)

print(
    "Intrinsic projection selection:    NOT ESTABLISHED"
)

print(
    "Exact frozen OPS selection:        NOT ESTABLISHED"
)

print("\nEND SIM17.3")






"""


RESULTS:



=================================================================
SIM17.3 — BILINEAR CONSTRAINTS AND SYMPLECTIC INHERITANCE
=================================================================

--- GATE A: CAYLEY-DICKSON MULTIPLICATION ---PASS | A signed multiplication index law
PASS | A identity and imaginary squares
PASS | A explicit nonassociativity
PASS | A exact bilinear structure constants

--- GATE B: 16D COMPLEX/SYMPLECTIC OPERATORS ---PASS | B all 15 imaginary left operators square to -I
PASS | B all 15 forms are symplectic and metric-compatible
PASS | B original octonion block formula
PASS | B doubled octonion block formula
PASS | B exactly 63 of 105 pairs anticommute
PASS | B exact nonanticommutation witness
B | pair partition: {'AA': 21, 'AB': 14, 'BB': 28}
B | nonanticommuting pairs: 42

--- GATE C: COORDINATE CARRIER CENSUS ---PASS | C all 6435 coordinate candidates considered
PASS | C exact 1+7+7 Clifford compatibility classes
PASS | C unique coordinate carrier with full ambient compatibility
PASS | C closure/projection/bilinear ranks separated
PASS | C seven quaternionic sectors per alternative carrier
C | compatibility classes: {'(True, 21, 21)': 1, '(True, 15, 21)': 7, '(False, 9, 9)': 7}

--- GATE D: 16D TO 8D TO 4D INHERITANCE ---PASS | D all 15 inherited forms classified on every 8-plane
PASS | D quaternionic four-plane Fano incidence ranks
PASS | D canonical carrier contains seven associative four-planes
PASS | D finite symplectic transformations and Lie generators
D | rank-8 inherited forms per carrier: 7
D | rank-4 inherited forms per quaternionic sector: 3

--- GATE E: COCHAIN AND CYCLE AUDIT ---PASS | E nontrivial associativity 3-coboundary
PASS | E normalized 3-coboundary
PASS | E 3-cocycle identity on all 65536 quadruples
PASS | E every quaternionic restriction has trivial coboundary
PASS | E all 315 translation controls give two C4 components

--- GATE F: PARENT NECESSITY ---PASS | F nonzero parent zero-divisor witness
PASS | F sixteen parent-algebra sign-character automorphisms
PASS | F sign-rephasing preserves exact Clifford compatibility
PASS | F graph inventory produced for all 15 carriers and both rules

--- GATE G: HELD-OUT OPS ACCEPTANCE ---G | cycle candidates: 0
G | exact labeled frozen C8: 0
Truth packet saved: SIM17.3_truth_packet.json
TRUTH_PACKET_SUMMARY={"adjacency_selected": false, "bilinear_law_intrinsically_selected": false, "canonical_selection_basis_dependent": true, "carrier_clifford_classes": [{"alternative": false, "ambient": 9, "count": 7, "restricted": 9}, {"alternative": true, "ambient": 15, "count": 7, "restricted": 21}, {"alternative": true, "ambient": 21, "count": 1, "restricted": 21}], "carrier_intrinsically_selected": false, "closed_coordinate_carriers": 15, "cocycle_quadruples_tested": 65536, "cocycle_triples_tested": 4096, "cycle_candidates": [], "distinguished_symplectic_form_selected": false, "exact_frozen_graph_matches": [], "frozen_p_h_signed_Omega_recovered": false, "full_OPS_derived": false, "individual_complex_structures": 15, "input_provenance": {"OPS": "Frozen edges, p, h and signed Omega introduced only in Gate G", "adjacency": "Two uniform exploratory extraction rules; neither privileged", "carrier_search": "All 6435 unital coordinate eight-planes; not all real eight-planes", "clifford": "Ambient left-multiplication anticommutators; distinct from SIM17.2 Cl(8,0)", "coordinates": "Given parent signed basis; not independently derived", "parent": "Fixed 16D Cayley-Dickson signed multiplication and Euclidean coordinate metric", "parent_symmetry": "Sixteen diagonal-character automorphisms; full parent group not classified", "physics": "No dynamical or physical mechanism claimed", "quaternions": "Derived from closure and associativity; no imported Fano sector list", "selection": "Complete ambient Clifford compatibility is an optional admissibility condition", "symplectic": "Fifteen parent forms -L_e_i; Euclidean metric supplied"}, "nontrivial_3_coboundary_triples": 1848, "parent_anticommuting_pairs": 63, "parent_anticommuting_partition": {"AA": 21, "AB": 14, "BB": 28}, "parent_sign_character_automorphisms": 16, "parent_zero_divisor_witness": {"x": [[1, 1], [10, -1]], "y": [[4, 1], [15, 1]]}, "projection_intrinsically_selected": false, "scope": "EXACT_RESTRICTED_COORDINATE_AUDIT", "sim": "17.3", "symplectic_eight_plane_rank_pattern": {"rank0": 8, "rank8": 7}, "symplectic_forms_on_parent": 15, "symplectic_quaternionic_rank_pattern": {"rank0": 12, "rank4": 3}, "unique_full_ambient_clifford_coordinate_carrier": [0, 1, 2, 3, 4, 5, 6, 7], "unsigned_translation_controls": 315, "vertex_set_intrinsically_selected": false}

=================================================================SIM17.3 — FINAL VALIDATOR VERDICT
=================================================================
16D multiplication:                 PASS
Individual complex structures:      15/15 PASS
Multiplication-derived symplectic:  15/15 PASS
Joint anticommutation:              63/105
Closed coordinate carriers:        15
Full ambient Clifford carrier:     UNIQUE IN CENSUS
Intrinsic projection selection:    NOT ESTABLISHED
Exact frozen OPS selection:        NOT ESTABLISHED

END SIM17.3
"""
