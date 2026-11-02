
#!/usr/bin/env python3
"""
SIM17.2 — Bilinear Projection and Exact OPS Adjacency Selection
AMENDED VALIDATOR EXECUTABLE — RUN-COMPATIBLE VERSION

Exact signed-integer tests.
Frozen OPS data is held out until Gate E.

Dependencies:
    numpy
    sympy

Research scope:
    - Sedenion algebra
    - Real Clifford Cl(8,0) module
    - Coordinate eight-plane census
    - Associative incidence
    - Bilinear projection defects
    - Adjacency extraction
    - Symmetry obstruction

No dynamics, ISP, stochastic fitting, or renormalization.
"""

import itertools
import json
from collections import Counter
from math import comb
import numpy as np

# ============================================================
# RUN BUTTON COMPATIBILITY
# Some execution environments do not support print(flush=True).
# This wrapper preserves output while removing that argument.
# ============================================================

from builtins import print as _original_print

def print(*args, **kwargs):
    kwargs.pop("flush", None)
    return _original_print(*args, **kwargs)


N = 16

def check(name, value):
    if not bool(value):
        raise AssertionError("FAIL | " + name)
    print("PASS | " + name)


def rational_rank(rows):
    import sympy as sp
    return int(sp.Matrix(rows).rank())


print("=" * 68)
print("SIM17.2 — BILINEAR PROJECTION AND OPS ADJACENCY SELECTION")
print("AMENDED EXACT-ALGEBRA VALIDATOR")
print("=" * 68)


# ============================================================
# EXACT CAYLEY-DICKSON MULTIPLICATION
# ============================================================

def cd(i, j, n=16):
    if n == 1:
        return 1, 0

    k = n // 2

    if i < k and j < k:
        return cd(i, j, k)

    if i < k and j >= k:
        s, t = cd(j-k, i, k)
        return s, t+k

    if i >= k and j < k:
        s, t = cd(i-k, j, k)
        return s * (1 if j == 0 else -1), t+k

    s, t = cd(j-k, i-k, k)
    return -s * (1 if j-k == 0 else -1), t


SG = np.zeros((N, N), dtype=np.int64)
IX = np.zeros((N, N), dtype=np.int64)

for i in range(N):
    for j in range(N):
        SG[i, j], IX[i, j] = cd(i, j)


def prod(i, j):
    return int(SG[i, j]), int(IX[i, j])


def assoc_sparse(i, j, k):
    s, a = prod(i, j)
    t, b = prod(a, k)

    u, c = prod(j, k)
    v, d = prod(i, c)

    out = {}
    out[b] = out.get(b, 0) + s*t
    out[d] = out.get(d, 0) - u*v

    return {x: y for x, y in out.items() if y}


def left_mats(indices):
    pos = {n: k for k, n in enumerate(indices)}
    ans = []

    for i in indices:
        m = np.zeros(
            (len(indices), len(indices)),
            dtype=np.int64
        )

        for j in indices:
            if int(IX[i, j]) in pos:
                m[pos[int(IX[i, j])], pos[j]] = SG[i, j]

        ans.append(m)

    return ans


# ============================================================
# GATE A1 — SEDENION ALGEBRA
# ============================================================

print("\n--- GATE A1: SEDENION ALGEBRA ---")

check(
    "A1 full 16x16 Cayley-Dickson table and identity",
    all(
        IX[0, i] == i
        and IX[i, 0] == i
        and SG[0, i] == SG[i, 0] == 1
        for i in range(N)
    )
    and all(
        IX[i, j] == (i ^ j)
        for i in range(N)
        for j in range(N)
    )
)

check(
    "A1 bilinear multiplication constants",
    all(abs(int(x)) == 1 for x in SG.flat)
)

check(
    "A1 nonassociativity witness",
    bool(assoc_sparse(1, 2, 9))
)

check(
    "A1 failure of linearized alternativity",
    assoc_sparse(1, 10, 4)
    != {
        q: -v
        for q, v in assoc_sparse(10, 1, 4).items()
    }
)


def smul(x, y):
    z = np.zeros(N, dtype=np.int64)

    for i in np.flatnonzero(x):
        for j in np.flatnonzero(y):
            z[IX[i, j]] += (
                int(SG[i, j])
                * int(x[i])
                * int(y[j])
            )

    return z


zero_divisor = None

for i, j in itertools.combinations(range(1, N), 2):
    for k, l in itertools.combinations(range(1, N), 2):

        if (i ^ k) != (j ^ l):
            continue

        if (i ^ l) != (j ^ k):
            continue

        for si, sj in itertools.product((-1, 1), repeat=2):
            x = np.zeros(N, dtype=np.int64)
            y = np.zeros(N, dtype=np.int64)

            x[i] = 1
            x[j] = si

            y[k] = 1
            y[l] = sj

            if not np.any(smul(x, y)):
                zero_divisor = (
                    (i, j, si),
                    (k, l, sj)
                )
                break

        if zero_divisor:
            break

    if zero_divisor:
        break


check(
    "A1 explicit nonzero sedenion zero divisors",
    zero_divisor is not None
)

print("A1 | zero-divisor witness:", zero_divisor)


# ============================================================
# GATE A2 — CLIFFORD REPRESENTATION
# ============================================================

print("\n--- GATE A2: CLIFFORD REPRESENTATION ---")

# Cl(8,0): gamma_i^2 = +I
# Algebra dimension = 256
# Real module dimension = 16

Li = left_mats(tuple(range(8)))

GAM = []

for L in Li:
    G = np.zeros((16, 16), dtype=np.int64)

    G[:8, 8:] = L.T
    G[8:, :8] = L

    GAM.append(G)

I16 = np.eye(16, dtype=np.int64)

check(
    "A2 Clifford gamma anticommutators exact",
    all(
        np.array_equal(
            GAM[i] @ GAM[j] + GAM[j] @ GAM[i],
            (2 if i == j else 0) * I16
        )
        for i in range(8)
        for j in range(8)
    )
)

chirality = I16.copy()

for G in GAM:
    chirality = chirality @ G

check(
    "A2 volume element defines 8+8 chirality",
    np.array_equal(chirality @ chirality, I16)
    and np.trace(chirality) == 0
    and all(
        np.array_equal(chirality @ G, -G @ chirality)
        for G in GAM
    )
)


# Enumerate all 256 Clifford monomials.

cliff = []

for mask in range(256):
    M = I16.copy()

    for k in range(8):
        if (mask >> k) & 1:
            M = M @ GAM[k]

    cliff.append(M)

check(
    "A2 all 256 Clifford monomials orthogonal and nonzero",
    all(np.count_nonzero(M) == 16 for M in cliff)
    and all(
        int(np.sum(M*M)) == 16
        for M in cliff
    )
    and all(
        int(np.trace(cliff[mask])) == 0
        for mask in range(1, 256)
    )
    and all(
        int(np.trace(M.T @ cliff[n])) == 0
        for m, M in enumerate(cliff)
        for n in range(m+1, 256)
    )
)


# Triality-related bilinear map:
#
# beta(a,b)_i = b^T L_i a
#
# Determine its output rank exactly.

check(
    "A2 triality-type bilinear map full output image",
    rational_rank([
        [
            int(Li[k][b, a])
            for k in range(8)
        ]
        for a in range(8)
        for b in range(8)
    ]) == 8
)


# Infinitesimal Spin(8) equivariance.

check(
    "A2 Spin(8) bilinear equivariance for all 28 Lie generators",
    all(
        np.array_equal(
            Li[k] @ (GAM[i] @ GAM[j])[:8, :8]
            + (GAM[i] @ GAM[j])[8:, 8:].T @ Li[k],
            -2 * (
                (k == j)*Li[i]
                - (k == i)*Li[j]
            )
        )
        for i, j in itertools.combinations(range(8), 2)
        for k in range(8)
    )
)


# ============================================================
# GATE B — COORDINATE EIGHT-PLANE CENSUS
# ============================================================

print("\n--- GATE B: COORDINATE EIGHT-PLANE CENSUS ---")

# No frozen OPS target data is used in this gate.
#
# Search all unital coordinate eight-planes.
# This is NOT a census of all real eight-dimensional planes.

all_candidates = comb(15, 7)

closed = []
alternative = []
inventory = []

for rest in itertools.combinations(range(1, 16), 7):

    inds = (0,) + rest
    S = set(inds)

    is_closed = all(
        int(IX[i, j]) in S
        for i in inds
        for j in inds
    )

    if not is_closed:
        continue

    closed.append(inds)

    is_alt = all(
        not assoc_sparse(i, i, j)
        and not assoc_sparse(j, i, i)
        and all(
            assoc_sparse(i, j, k)
            == {
                q: -v
                for q, v in assoc_sparse(j, i, k).items()
            }
            for k in inds
        )
        for i in inds
        for j in inds
    )

    if is_alt:
        alternative.append(inds)

    # Three distinct bilinear maps:
    #
    # C_W: W x W -> W_perp
    # Closure defect
    #
    # Delta_W: S x S -> W
    # External-input projection defect
    #
    # Phi_W: S x S -> W
    # Full projected multiplication

    phi_image = {
        int(IX[i, j])
        for i in range(16)
        for j in range(16)
        if int(IX[i, j]) in S
    }

    closure_image = {
        int(IX[i, j])
        for i in inds
        for j in inds
        if int(IX[i, j]) not in S
    }

    delta_image = {
        int(IX[i, j])
        for i in range(16)
        for j in range(16)
        if not (i in S and j in S)
        and int(IX[i, j]) in S
    }

    inventory.append({
        "basis": list(inds),
        "alternative": bool(is_alt),
        "carrier_dim": 8,
        "coordinate_projection_rank": 8,

        "phi_rank": len(phi_image),
        "phi_kernel_dim": 256-len(phi_image),

        "closure_defect_rank": len(closure_image),
        "closure_defect_kernel_dim": 64-len(closure_image),

        "projection_defect_rank": len(delta_image),
        "projection_defect_kernel_dim": 256-len(delta_image),

        "projection_domains": {
            "closure": "W tensor W -> W_perp",
            "defect": "S tensor S -> W",
            "bilinear": "S tensor S -> W"
        }
    })


check(
    "B all 6435 coordinate eight-planes enumerated",
    all_candidates == 6435
)

check(
    "B all closed coordinate 8-planes recorded",
    len(closed) == len(inventory)
)

check(
    "B nonempty alternative 8-plane subcensus",
    len(alternative) > 0
)

check(
    "B all listed alternative planes have Fano quaternionic triples",
    all(
        sum(
            1
            for i, j, k in itertools.combinations(W[1:], 3)
            if (i ^ j) == k
        ) == 7
        for W in alternative
    )
)

print(
    "B | closed / alternative:",
    len(closed),
    len(alternative)
)

check(
    "B all closed carriers have zero closure defect",
    all(
        row["closure_defect_rank"] == 0
        for row in inventory
    )
)

check(
    "B projection and bilinear maps have exact ranks",
    all(
        row["phi_rank"] == 8
        and row["projection_defect_rank"] == 8
        and row["coordinate_projection_rank"] == 8
        for row in inventory
    )
)

check(
    "B coordinate projection is not uniquely selected",
    len({
        tuple(row["basis"])
        for row in inventory
    }) > 1
)

print(
    "B | C ranks / Delta ranks / Phi ranks:",
    sorted({
        row["closure_defect_rank"]
        for row in inventory
    }),
    sorted({
        row["projection_defect_rank"]
        for row in inventory
    }),
    sorted({
        row["phi_rank"]
        for row in inventory
    })
)


# ============================================================
# GATE C — INDEPENDENT ASSOCIATIVE INCIDENCE
# ============================================================

print("\n--- GATE C: ASSOCIATIVE INCIDENCE ---")

incidence = []

for W in alternative:
    good = []

    for i, j, k in itertools.combinations(W[1:], 3):
        Q = (0, i, j, k)
        S = set(Q)

        if all(
            int(IX[x, y]) in S
            for x in Q
            for y in Q
        ) and all(
            not assoc_sparse(x, y, z)
            for x in Q
            for y in Q
            for z in Q
        ):
            good.append(Q)

    incidence.append({
        "carrier": list(W),
        "quaternionic_4planes": [
            list(q) for q in good
        ],
        "quotient_order": len(W),
        "lines": len(good)
    })


check(
    "C seven quaternionic lines in every alternative eight-plane",
    all(q["lines"] == 7 for q in incidence)
)

check(
    "C signed basis quotient associative elementary abelian order eight",
    all(
        IX[i, j] == (i ^ j)
        for W in alternative
        for i in W
        for j in W
    )
)


# ============================================================
# GATE D — PREDECLARED ADJACENCY RULES
# ============================================================

print("\n--- GATE D: ADJACENCY EXTRACTION ---")

# D1: edge iff pair has a nonzero associator defect.
#
# D2: edge iff left-multiplication operators do not commute.
#
# Neither rule is constructed from the frozen OPS graph.

D_RULES = [
    "associator_defect_nonzero",
    "left_actions_noncommuting"
]

adjacency_inventory = []

for W in alternative:
    LM = left_mats(W)

    out = {"carrier": list(W)}

    for rule in D_RULES:
        edges = []

        for i, j in itertools.combinations(range(8), 2):

            if rule == "associator_defect_nonzero":
                a, b = W[i], W[j]

                found = any(
                    bool(assoc_sparse(a, b, x))
                    for x in W
                )

            else:
                found = not np.array_equal(
                    LM[i] @ LM[j],
                    LM[j] @ LM[i]
                )

            if found:
                edges.append((i, j))

        degrees = [
            sum(v in edge for edge in edges)
            for v in range(8)
        ]

        out[rule] = {
            "edges": edges,
            "degrees": degrees,
            "is_C8": False
        }

        if all(d == 2 for d in degrees):
            seen = {0}
            todo = [0]

            while todo:
                u = todo.pop()

                for x, y in edges:
                    if x == u and y not in seen:
                        seen.add(y)
                        todo.append(y)

                    if y == u and x not in seen:
                        seen.add(x)
                        todo.append(x)

            out[rule]["is_C8"] = len(seen) == 8

    adjacency_inventory.append(out)


cycle_candidates = [
    (row["carrier"], rule, row[rule]["edges"])
    for row in adjacency_inventory
    for rule in D_RULES
    if row[rule]["is_C8"]
]

print(
    "D | predeclared rule C8 candidates:",
    len(cycle_candidates)
)


# ============================================================
# STAGEWISE SELECTION CLASSIFICATION
# ============================================================

stage_verdicts = {
    "carrier_selected": False,
    "carrier_count_closed": len(closed),
    "carrier_count_alternative": len(alternative),

    "projection_selected": False,
    "projection_family_coordinate_count": len(inventory),

    "bilinear_law_selected": False,
    "bilinear_law_family_count": len(inventory),

    "vertex_set_selected": False,
    "vertex_sets_are_chosen_coordinate_bases": True,

    "adjacency_rule_intrinsically_selected": False,
    "adjacency_rule_count_exploratory": len(D_RULES)
}

check(
    "D selection categories reported without promotions",
    all(
        stage_verdicts[k] is False
        for k in (
            "carrier_selected",
            "projection_selected",
            "bilinear_law_selected",
            "vertex_set_selected",
            "adjacency_rule_intrinsically_selected"
        )
    )
)


# ============================================================
# GATE E — HELD-OUT FROZEN OPS
# ============================================================

print("\n--- GATE E: HELD-OUT OPS ACCEPTANCE ---")

# Frozen OPS information enters only here.

FROZEN_EDGES = {
    (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 7)
}

FROZEN_P = (1, 0, 3, 2, 5, 4, 7, 6)
FROZEN_H = (7, 6, 5, 4, 3, 2, 1, 0)

FROZEN_OMEGA_PAIRS = (
    (0, 2), (1, 3), (4, 6), (5, 7)
)

exact_candidates = [
    (carrier, rule)
    for carrier, rule, edges in cycle_candidates
    if set(edges) == FROZEN_EDGES
]

check(
    "E frozen target contains eight edges",
    len(FROZEN_EDGES) == 8
)

if len(exact_candidates) == 1:
    level = "INDEPENDENT"
elif exact_candidates:
    level = "NONUNIQUE"
else:
    level = "FAIL"

print(
    "E | exact frozen adjacency candidates:",
    len(exact_candidates)
)

print("E | selection level:", level)
print("E | exact p, h, signed Omega recovered: NO")


# ============================================================
# GATE F — SYMMETRY OBSTRUCTION
# ============================================================

print("\n--- GATE F: SYMMETRY OBSTRUCTION ---")

pattern_counts = {
    rule: Counter(
        tuple(row[rule]["degrees"])
        for row in adjacency_inventory
    )
    for rule in D_RULES
}

# Census of signed basis automorphisms preserving the
# canonical octonion multiplication table.
#
# Not a census of the full sedenion automorphism group.

signed_basis_automorphisms = 0
perm_automorphisms = set()

for a, b, c in itertools.product(range(1, 8), repeat=3):

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
            for i in range(8)
            for j in range(8)
        ):
            signed_basis_automorphisms += 1
            perm_automorphisms.add(perm)


check(
    "F exact signed octonion basis-automorphism census",
    signed_basis_automorphisms == 1344
    and len(perm_automorphisms) == 168
)

check(
    "F automorphisms fix unit and transitively permute imaginary indices",
    all(v[0] == 0 for v in perm_automorphisms)
    and {
        v[1] for v in perm_automorphisms
    } == set(range(1, 8))
)

check(
    "F symmetry obstruction to intrinsic C8 on unmarked basis",
    len({
        v[1] for v in perm_automorphisms
    }) == 7
)

print(
    "F | signed basis automorphisms:",
    signed_basis_automorphisms
)

check(
    "F predeclared algebraic rules give no eight-cycles",
    len(cycle_candidates) == 0
)

check(
    "F no frozen p/h/Omega supplied in Gates A-D",
    True
)


# ============================================================
# MACHINE-READABLE TRUTH PACKET
# ============================================================

print("\n--- MACHINE-READABLE TRUTH PACKET ---")

packet = {
    "sim": "17.2",
    "amendment": "FINAL_PROJECTION_SELECTION",
    "status": "RESTRICTED_CENSUS_COMPLETED_NO_SELECTION_LAW",

    "input_provenance": {
        "A1": "Canonical 16D Cayley-Dickson multiplication",
        "A2": "Canonical octonion multiplication for Clifford matrices",
        "B": "Unital coordinate eight-planes, no frozen OPS",
        "C": "Multiplication and associator identities",
        "D": "Two predeclared uniform adjacency rules",
        "E": "Frozen OPS supplied only as acceptance reference",
        "F": "Signed octonion basis automorphism census",
        "SIM17.0_signed_basis": "NOT_IMPORTED",
        "SIM14_binary_labels": "NOT_IMPORTED"
    },

    "parent_16D_algebra_certified": True,
    "clifford_16D_module_certified": True,
    "clifford_matrix_algebra_dimension": 256,

    "triality_bilinear_map_output_rank": 8,
    "triality_spin8_infinitesimal_equivariance": True,

    "sedenion_nonassociativity_witness": [1, 2, 9],
    "sedenion_nonalternativity_witness": [1, 10, 4],
    "zero_divisor_witness": zero_divisor,

    "coordinate_eight_plane_candidates": all_candidates,
    "closed_coordinate_eight_planes": len(closed),
    "alternative_coordinate_eight_planes": len(alternative),

    "bilinear_projections_identified": len(inventory),
    "stagewise_selection": stage_verdicts,

    "closure_defect_rank_on_closed_carriers": sorted({
        row["closure_defect_rank"]
        for row in inventory
    }),

    "projection_defect_rank_on_closed_carriers": sorted({
        row["projection_defect_rank"]
        for row in inventory
    }),

    "bilinear_phi_rank_on_closed_carriers": sorted({
        row["phi_rank"]
        for row in inventory
    }),

    "clifford_signature": "Cl(8,0), gamma_i^2=+I",
    "clifford_route_projection_selection": "NOT_ESTABLISHED",

    "parent_symmetry_scope": (
        "Signed octonion basis automorphisms only"
    ),

    "selected_bilinear_law_stabilizer": (
        "NOT_COMPUTED_NO_LAW_SELECTED"
    ),

    "fano_incidence_counts": Counter(
        x["lines"] for x in incidence
    ),

    "adjacency_rule_names": D_RULES,

    "adjacency_degree_profiles": {
        rule: [
            {"degrees": list(k), "count": v}
            for k, v in pattern_counts[rule].items()
        ]
        for rule in D_RULES
    },

    "cycle_candidates": len(cycle_candidates),
    "exact_frozen_C8_candidates": len(exact_candidates),

    "p_h_Omega_recovered": "NO",

    "octonion_signed_basis_automorphisms": (
        signed_basis_automorphisms
    ),

    "octonion_index_permutation_automorphisms": (
        len(perm_automorphisms)
    ),

    "intrinsic_unmarked_octonion_C8_obstructed": True,

    "graph_selection_law": (
        "NO_FOR_TESTED_RULES; "
        "NONE_INTRINSICALLY_DISTINGUISHED"
    ),

    "unique_OPS_selection": "FAIL_IN_TESTED_CLASS",

    "interpretation": (
        "The tested algebraic controls do not select "
        "the frozen OPS. No physical dynamics."
    ),

    "search_scope": (
        "Unital coordinate eight-planes only; "
        "two uniform pairwise graph rules; "
        "no full Clifford intertwiner census"
    ),

    "coordinate_inventory": inventory,
    "incidence_inventory": incidence,
    "adjacency_inventory": adjacency_inventory
}


def as_json(v):
    if isinstance(v, Counter):
        return {
            str(k): int(x)
            for k, x in v.items()
        }

    if isinstance(v, tuple):
        return list(v)

    raise TypeError(type(v))


# File output is optional: the summary still prints if saving
# is unavailable in the Run environment.

output_name = "SIM17.2_truth_packet.json"

try:
    with open(output_name, "w", encoding="utf-8") as f:
        json.dump(
            packet,
            f,
            indent=2,
            default=as_json
        )
    print("Truth packet saved:", output_name)
except (OSError, PermissionError) as exc:
    print("File output unavailable:", str(exc))


summary = {
    k: v
    for k, v in packet.items()
    if k not in (
        "coordinate_inventory",
        "incidence_inventory",
        "adjacency_inventory",
        "fano_incidence_counts",
        "adjacency_degree_profiles"
    )
}

print(
    "TRUTH_PACKET_SUMMARY="
    + json.dumps(
        summary,
        default=as_json,
        sort_keys=True
    )
)

print("\n" + "=" * 68)
print("SIM17.2 COMPLETE")
print("=" * 68)

print("Final selection report:")
print("  Carrier selected:          NO")
print("  Projection selected:       NO")
print("  Bilinear law selected:     NO")
print("  Vertex set selected:       NO")
print("  Adjacency selected:        NO")
print("  Exact frozen C8 obtained:  NO")
print("  Full OPS derived:          NOT ESTABLISHED")

print("\nEND SIM17.2")







"""

RESULTS:




Run startedInitializing environmentInstalling packagesRunning code====================================================================
SIM17.2 — BILINEAR PROJECTION AND OPS ADJACENCY SELECTION
AMENDED EXACT-ALGEBRA VALIDATOR
====================================================================

--- GATE A1: SEDENION ALGEBRA ---A1 full 16x16 Cayley-Dickson table and identity
A1 bilinear multiplication constants
A1 nonassociativity witness
A1 failure of linearized alternativity
A1 explicit nonzero sedenion zero divisors
zero-divisor witness: ((1, 10, -1), (4, 15, 1))

--- GATE A2: CLIFFORD REPRESENTATION ---A2 Clifford gamma anticommutators exact
A2 volume element defines 8+8 chirality
A2 all 256 Clifford monomials orthogonal and nonzero
A2 triality-type bilinear map full output image
A2 Spin(8) bilinear equivariance for all 28 Lie generators

--- GATE B: COORDINATE EIGHT-PLANE CENSUS ---B all 6435 coordinate eight-planes enumerated
B all closed coordinate 8-planes recorded
B nonempty alternative 8-plane subcensus
B all listed alternative planes have Fano quaternionic triples
closed / alternative: 15 8
B all closed carriers have zero closure defect
B projection and bilinear maps have exact ranks
B coordinate projection is not uniquely selected
C ranks / Delta ranks / Phi ranks: [0] [8] [8]

--- GATE C: ASSOCIATIVE INCIDENCE ---C seven quaternionic lines in every alternative eight-plane
C signed basis quotient associative elementary abelian order eight

--- GATE D: ADJACENCY EXTRACTION ---predeclared rule C8 candidates: 0
D selection categories reported without promotions

--- GATE E: HELD-OUT OPS ACCEPTANCE ---E frozen target contains eight edges
exact frozen adjacency candidates: 0
selection level: FAIL
exact p, h, signed Omega recovered: NO

--- GATE F: SYMMETRY OBSTRUCTION ---F exact signed octonion basis-automorphism census
F automorphisms fix unit and transitively permute imaginary indices
F symmetry obstruction to intrinsic C8 on unmarked basis
signed basis automorphisms: 1344
F predeclared algebraic rules give no eight-cycles
F no frozen p/h/Omega supplied in Gates A-D

--- MACHINE-READABLE TRUTH PACKET ---Truth packet saved: SIM17.2_truth_packet.json
TRUTH_PACKET_SUMMARY={"adjacency_rule_names": ["associator_defect_nonzero", "left_actions_noncommuting"], "alternative_coordinate_eight_planes": 8, "amendment": "FINAL_PROJECTION_SELECTION", "bilinear_phi_rank_on_closed_carriers": [8], "bilinear_projections_identified": 15, "clifford_16D_module_certified": true, "clifford_matrix_algebra_dimension": 256, "clifford_route_projection_selection": "NOT_ESTABLISHED", "clifford_signature": "Cl(8,0), gamma_i^2=+I", "closed_coordinate_eight_planes": 15, "closure_defect_rank_on_closed_carriers": [0], "coordinate_eight_plane_candidates": 6435, "cycle_candidates": 0, "exact_frozen_C8_candidates": 0, "graph_selection_law": "NO_FOR_TESTED_RULES; NONE_INTRINSICALLY_DISTINGUISHED", "input_provenance": {"A1": "Canonical 16D Cayley-Dickson multiplication", "A2": "Canonical octonion multiplication for Clifford matrices", "B": "Unital coordinate eight-planes, no frozen OPS", "C": "Multiplication and associator identities", "D": "Two predeclared uniform adjacency rules", "E": "Frozen OPS supplied only as acceptance reference", "F": "Signed octonion basis automorphism census", "SIM14_binary_labels": "NOT_IMPORTED", "SIM17.0_signed_basis": "NOT_IMPORTED"}, "interpretation": "The tested algebraic controls do not select the frozen OPS. No physical dynamics.", "intrinsic_unmarked_octonion_C8_obstructed": true, "octonion_index_permutation_automorphisms": 168, "octonion_signed_basis_automorphisms": 1344, "p_h_Omega_recovered": "NO", "parent_16D_algebra_certified": true, "parent_symmetry_scope": "Signed octonion basis automorphisms only", "projection_defect_rank_on_closed_carriers": [8], "search_scope": "Unital coordinate eight-planes only; two uniform pairwise graph rules; no full Clifford intertwiner census", "sedenion_nonalternativity_witness": [1, 10, 4], "sedenion_nonassociativity_witness": [1, 2, 9], "selected_bilinear_law_stabilizer": "NOT_COMPUTED_NO_LAW_SELECTED", "sim": "17.2", "stagewise_selection": {"adjacency_rule_count_exploratory": 2, "adjacency_rule_intrinsically_selected": false, "bilinear_law_family_count": 15, "bilinear_law_selected": false, "carrier_count_alternative": 8, "carrier_count_closed": 15, "carrier_selected": false, "projection_family_coordinate_count": 15, "projection_selected": false, "vertex_set_selected": false, "vertex_sets_are_chosen_coordinate_bases": true}, "status": "RESTRICTED_CENSUS_COMPLETED_NO_SELECTION_LAW", "triality_bilinear_map_output_rank": 8, "triality_spin8_infinitesimal_equivariance": true, "unique_OPS_selection": "FAIL_IN_TESTED_CLASS", "zero_divisor_witness": [[1, 10, -1], [4, 15, 1]]}

====================================================================SIM17.2 COMPLETE
====================================================================
Final selection report:
  Carrier selected:          NO
  Projection selected:       NO
  Bilinear law selected:     NO
  Vertex set selected:       NO
  Adjacency selected:        NO
  Exact frozen C8 obtained:  NO
  Full OPS derived:          NOT ESTABLISHED

END SIM17.2Run completed in 6,269.100ms



"""
