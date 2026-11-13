
#!/usr/bin/env python3
"""SIM17.5 precursor: exact SIM14.8 / SIM17.4 order-192 group bridge audit.

Reconstructs directly from the declared finite inputs:
  * SIM14.8 affine partner group, in binary coordinates recoded so the
    distinguished vector 001 is 100 (integer 4).
  * SIM17.4 signed octonion-basis automorphisms preserving the DECLARED
    octavian half-unit support code.
  * Each group's order, center, quotient, normal eight-element module.
  * The quotient's direct action on binary translations and contragredient
    action on sign characters, including a perfect invariant pairing.
  * The 1+6 versus 3+4 orbit partitions, and their quaternionic-line incidence.

Only Python 3 standard library. No NumPy, SymPy, external files, or network.
No OPS reconstruction, root-selection, or physical-dynamics claims.

Reference source paths (GitHub repository RelationalCoherenceFieldTheory_RCFT):
  Projection_Dynamics/Operational_Phase_Space/SIM14.8.py
  Projection_Dynamics/Operational_Phase_Space/SIM17.4.py
"""

from itertools import combinations, product
from collections import Counter
import json

V = range(8)
ID = tuple(V)
ONE_SIGNS = (1,) * 8


def par(a):
    return (a.bit_count() & 1)


def dot(t, x):
    """Standard F2-bilinear pairing between covectors and vectors."""
    return par(t & x)


def character(t):
    return tuple((-1 if dot(t, x) else 1) for x in V)


def cd(i, j, n=8):
    """SIM17.4 Cayley-Dickson signed basis product e_i e_j."""
    if n == 1:
        return (1, 0)
    k = n // 2
    if i < k and j < k:
        return cd(i, j, k)
    if i < k and j >= k:
        s, t = cd(j - k, i, k)
        return s, t + k
    if i >= k and j < k:
        s, t = cd(i - k, j, k)
        return s * (1 if j == 0 else -1), t + k
    s, t = cd(j - k, i - k, k)
    return -s * (1 if j - k == 0 else -1), t


SG = tuple(tuple(cd(i, j)[0] for j in V) for i in V)
assert all(cd(i, j)[1] == (i ^ j) for i, j in product(V, repeat=2))

CONVERT = (0, 4, 1, 5, 7, 3, 6, 2)
OLD_SUPPORTS = (
    (0, 1, 2, 3), (0, 1, 4, 5), (0, 1, 6, 7),
    (0, 2, 4, 7), (0, 2, 5, 6), (0, 3, 4, 6),
    (0, 3, 5, 7), (1, 2, 4, 6), (1, 2, 5, 7),
    (1, 3, 4, 7), (1, 3, 5, 6), (2, 3, 4, 5),
    (2, 3, 6, 7), (4, 5, 6, 7),
)
SUPPORTS = {frozenset(CONVERT[i] for i in s) for s in OLD_SUPPORTS}
assert len(SUPPORTS) == 14


def linear_perm(a, b, c):
    """Map x to (x0*a XOR x1*b XOR x2*c) with bit order 1,2,4."""
    return tuple(
        (a if (x & 1) else 0) ^
        (b if (x & 2) else 0) ^
        (c if (x & 4) else 0)
        for x in V
    )


GL = {
    linear_perm(a, b, c)
    for a, b, c in product(range(1, 8), repeat=3)
    if b != a and c not in (a, b, a ^ b)
}
assert len(GL) == 168


def pcompose(p, q):
    """Permutation composition p after q."""
    return tuple(p[q[i]] for i in V)


def pinverse(p):
    result = [0] * 8
    for i, j in enumerate(p):
        result[j] = i
    return tuple(result)


# The 24-element quotient from SIM17.4 is a stabilizer of direction 4.
Q = {p for p in GL if p[4] == 4}
assert len(Q) == 24


# Enumerate signed octonion automorphisms (no group identification assumed).
AUTOS = set()
for perm in GL:
    for mask in range(128):
        signs = (1,) + tuple(
            -1 if (mask >> (i - 1)) & 1 else 1 for i in range(1, 8)
        )
        if all(
            SG[i][j] * signs[i ^ j] ==
            signs[i] * signs[j] * SG[perm[i]][perm[j]]
            for i, j in product(V, repeat=2)
        ):
            AUTOS.add((perm, signs))
assert len(AUTOS) == 1344


def preserves_supports(perm):
    return {frozenset(perm[i] for i in s) for s in SUPPORTS} == SUPPORTS


H17 = {(perm, signs) for perm, signs in AUTOS if preserves_supports(perm)}
Q17 = {p for p, _ in H17}
assert len(H17) == 192 and len(Q17) == 24
assert Q17 == Q
assert len({s for p, s in H17 if p == ID}) == 8
assert {s for p, s in H17 if p == ID} == {character(t) for t in V}
assert Counter(p for p, _ in H17) == Counter({p: 8 for p in Q})


def scompose(g, h):
    """Compose signed-basis transformations: g after h."""
    pg, sg = g
    ph, sh = h
    return (
        tuple(pg[ph[i]] for i in V),
        tuple(sh[i] * sg[ph[i]] for i in V),
    )


def sinverse(g):
    p, s = g
    ip = pinverse(p)
    return ip, tuple(s[ip[i]] for i in V)


SIGNED_ID = (ID, ONE_SIGNS)
assert SIGNED_ID in H17
for g in H17:
    assert scompose(g, sinverse(g)) == SIGNED_ID
    assert scompose(sinverse(g), g) == SIGNED_ID
for g, h in product(H17, repeat=2):
    assert scompose(g, h) in H17

Z17 = {
    g for g in H17
    if all(scompose(g, h) == scompose(h, g) for h in H17)
}
assert Z17 == {SIGNED_ID}


# SIM14.8 affine partner group in a common binary basis.
# In SIM14.8 s_Omega is 001. The involutive bit-reversal below sends
# 001 -> 100 (integer 4), so the quotient is exactly Q above.

def swap_bits_1_4(x):
    return ((x & 1) << 2) | (x & 2) | ((x & 4) >> 2)

S = tuple(swap_bits_1_4(x) for x in V)
Q14_original = {p for p in GL if p[1] == 1}
Q14_recoded = {pcompose(S, pcompose(p, S)) for p in Q14_original}
assert Q14_recoded == Q

# In SIM17.0's explicit label conversion (also reused in SIM17.4),
# the frozen SIM14 permutations are translations by DIFFERENT vectors.
# This is a held-out comparison, not part of the upstream selection.
SIM14_P = (1, 0, 3, 2, 5, 4, 7, 6)
SIM14_H = (7, 6, 5, 4, 3, 2, 1, 0)
SIM14_SOMEGA = (2, 3, 0, 1, 6, 7, 4, 5)


def converted_translation(g):
    diffs = {CONVERT[i] ^ CONVERT[g[i]] for i in V}
    assert len(diffs) == 1
    return next(iter(diffs))


OPS_DIRECTIONS_IN_CD = {
    'p': converted_translation(SIM14_P),
    'h': converted_translation(SIM14_H),
    's_omega': converted_translation(SIM14_SOMEGA),
}
assert OPS_DIRECTIONS_IN_CD == {'p': 4, 'h': 2, 's_omega': 1}

# Affine elements (translation v, linear permutation p): x -> p(x) XOR v.
H14 = {(v, p) for v in V for p in Q}


def acompose(g, h):
    """Affine composition g after h."""
    vg, pg = g
    vh, ph = h
    return (vg ^ pg[vh], pcompose(pg, ph))


assert len(H14) == 192
for g, h in product(H14, repeat=2):
    assert acompose(g, h) in H14
Z14 = {
    g for g in H14
    if all(acompose(g, h) == acompose(h, g) for h in H14)
}
assert Z14 == {(0, ID), (4, ID)}


# Action of each quotient permutation p on a dual character t.
# chi_t transforms to chi_t circ p^{-1}.
CHAR_LOOKUP = {character(t): t for t in V}
assert len(CHAR_LOOKUP) == 8


def dual_action(p, t):
    ip = pinverse(p)
    transformed = tuple(character(t)[ip[x]] for x in V)
    return CHAR_LOOKUP[transformed]


# This is genuinely the signed automorphism group's conjugation action
# on its normal 8-element character subgroup, not just a guessed dual action.
for g in H17:
    p, _ = g
    for t in V:
        chi = (ID, character(t))
        conj = scompose(scompose(g, chi), sinverse(g))
        assert conj == (ID, character(dual_action(p, t)))

# The natural pairing is invariant under the *paired* actions.
for p, t, x in product(Q, V, V):
    assert dot(dual_action(p, t), p[x]) == dot(t, x)

fixed_vectors = sorted(x for x in V if all(p[x] == x for p in Q))
fixed_characters = sorted(t for t in V if all(dual_action(p, t) == t for p in Q))
assert fixed_vectors == [0, 4]
assert fixed_characters == [0]


def orbit_partition(actions, points):
    remaining = set(points)
    answer = []
    while remaining:
        x = min(remaining)
        orbit = {f(x) for f in actions}
        assert orbit <= remaining
        remaining -= orbit
        answer.append(sorted(orbit))
    return sorted(answer, key=lambda o: (len(o), o))


vector_orbits = orbit_partition([lambda x, p=p: p[x] for p in Q], range(1, 8))
character_orbits = orbit_partition([lambda t, p=p: dual_action(p, t) for p in Q], range(1, 8))
assert vector_orbits == [[4], [1, 2, 3, 5, 6, 7]]
assert character_orbits == [[1, 2, 3], [4, 5, 6, 7]]


# Seven Fano quaternionic 4D coordinate carriers, indexed by 2D
# subspaces of F2^3. Each such line is the kernel of a unique nonzero
# character. This gives a canonical Q-equivariant 3+4 partition.
FANO_PLANES = {
    frozenset((0, a, b, a ^ b))
    for a, b in combinations(range(1, 8), 2)
}
assert len(FANO_PLANES) == 7
CHAR_KERNELS = {
    t: frozenset(x for x in V if dot(t, x) == 0)
    for t in range(1, 8)
}
assert set(CHAR_KERNELS.values()) == FANO_PLANES
assert len(CHAR_KERNELS) == 7
for p, t in product(Q, range(1, 8)):
    lhs = frozenset(p[x] for x in CHAR_KERNELS[t])
    rhs = CHAR_KERNELS[dual_action(p, t)]
    assert lhs == rhs

# The DECLARED octavian code is precisely what singles out three of the
# seven quaternionic planes as containing half-integral elements.
rich_planes = FANO_PLANES & SUPPORTS
plain_planes = FANO_PLANES - SUPPORTS
assert len(rich_planes) == 3 and len(plain_planes) == 4
assert rich_planes == {F for F in FANO_PLANES if 4 in F}
assert plain_planes == {F for F in FANO_PLANES if 4 not in F}
assert {t for t, F in CHAR_KERNELS.items() if F in rich_planes} == {1, 2, 3}
assert {t for t, F in CHAR_KERNELS.items() if F in plain_planes} == {4, 5, 6, 7}

# Independently count norm-one octavian units in each 4D associative
# coordinate sector using the DECLARED 240-root construction.
ROOTS2 = set()  # coordinates of 2*x
for i in V:
    for sgn in (-2, 2):
        x = [0] * 8
        x[i] = sgn
        ROOTS2.add(tuple(x))
for support in SUPPORTS:
    for signs in product((-1, 1), repeat=4):
        x = [0] * 8
        for i, sgn in zip(sorted(support), signs):
            x[i] = sgn
        ROOTS2.add(tuple(x))
assert len(ROOTS2) == 240
ROOT_COUNT_BY_PLANE = {
    tuple(sorted(F)): sum(
        all(x[i] == 0 for i in V if i not in F)
        for x in ROOTS2
    )
    for F in FANO_PLANES
}
assert {ROOT_COUNT_BY_PLANE[tuple(sorted(F))] for F in rich_planes} == {24}
assert {ROOT_COUNT_BY_PLANE[tuple(sorted(F))] for F in plain_planes} == {8}

# Quotient acts faithfully on the four planes not containing the fixed
# direction: this exhibits Q as S4 via a faithful four-point action.
plain = sorted(plain_planes, key=lambda f: tuple(sorted(f)))
permutations_of_four = {
    tuple(plain.index(frozenset(p[x] for x in F)) for F in plain)
    for p in Q
}
assert len(permutations_of_four) == 24

# No Q-equivariant linear identification V -> V* is possible: a module
# isomorphism must preserve the dimensions of fixed subspaces, but these
# have dimensions 1 and 0, respectively. The invariant pairing is between
# DIFFERENT dual modules; this is not an isomorphism of the modules.
assert len(fixed_vectors) != len(fixed_characters)

# Canonical alternating form on the 6-dimensional BINARY direct sum
# V + V*: omega((x,t),(y,u)) = t(y) + u(x) over F2.
# This is NOT the frozen real 8D SIM14 Omega. It is an optional finite
# symplectic invariant constructed from the duality of the two modules.
def omega_binary(a, b):
    x, t = a
    y, u = b
    return dot(t, y) ^ dot(u, x)


E6 = tuple(product(V, V))
assert len(E6) == 64
assert all(omega_binary(a, a) == 0 for a in E6)
assert all(any(omega_binary(a, b) for b in E6) for a in E6 if a != (0, 0))
assert all(
    omega_binary((p[x], dual_action(p, t)),
                 (p[y], dual_action(p, u))) ==
    omega_binary((x, t), (y, u))
    for p in Q for x, t in E6 for y, u in E6
)

results = {
    'source': 'SIM14.8 / SIM17.4 independently reconstructed finite inputs',
    'count_signed_basis_automorphisms': len(AUTOS),
    'count_octavian_lattice_preserving_signed_automorphisms': len(H17),
    'quotient_order': len(Q),
    'quotient_is_S4_via_four_plane_action': len(permutations_of_four) == 24,
    'center_SIM14_affine_order': len(Z14),
    'center_SIM17_signed_order': len(Z17),
    'groups_isomorphic': False,
    'common_fixed_vectors_mod2': fixed_vectors,
    'common_fixed_characters_mod2': fixed_characters,
    'primal_nonzero_direction_orbits': vector_orbits,
    'dual_nonzero_character_orbits': character_orbits,
    'dual_pairing_Q_invariant': True,
    'canonical_binary_symplectic_6D_Q_invariant': True,
    'held_out_OPS_directions_in_SIM17_coordinate_map': OPS_DIRECTIONS_IN_CD,
    'normal_modules_Q_equivariantly_isomorphic': False,
    'associative_plane_count': len(FANO_PLANES),
    'planes_with_half_integral_octavian_support': [sorted(x) for x in sorted(rich_planes, key=lambda f: tuple(sorted(f)))],
    'planes_without_half_integral_octavian_support': [sorted(x) for x in sorted(plain_planes, key=lambda f: tuple(sorted(f)))],
    'arithmetic_plane_split': [len(rich_planes), len(plain_planes)],
    'octavian_norm_one_unit_counts_in_fano_planes': {str(k): v for k, v in ROOT_COUNT_BY_PLANE.items()},
    'selection_of_octavian_order_from_parent': 'NOT_TESTED',
    'independent_frozen_OPS_reconstruction': False,
}

print('SIM17 / SIM14 ORDER-192 DUAL-MODULE AUDIT: ALL ASSERTIONS PASSED')
print('SIM14 affine: |H|=192, |Z(H)|=', len(Z14))
print('SIM17 signed: |H|=192, |Z(H)|=', len(Z17))
print('Shared quotient |Q|=', len(Q), ' acts faithfully on four Fano sectors (S4)')
print('Common fixed vectors:', fixed_vectors)
print('Common fixed characters:', fixed_characters)
print('Vector orbits (nonzero):', vector_orbits)
print('Dual character orbits (nonzero):', character_orbits)
print('Octavian support-rich planes:', [sorted(x) for x in sorted(rich_planes, key=lambda f: tuple(sorted(f)))])
print('Root counts in rich/plain quaternionic sectors: 24 / 8 (3+4 sectors)')
print('Q-invariant pairing <t,x> = parity(t & x): PASS')
print('Canonical binary alternating form on V+V*: PASS (NOT frozen real Omega)')
print('Held-out SIM14 directions in SIM17 coordinate map:', OPS_DIRECTIONS_IN_CD)
print('Parent root-system selection and frozen OPS reconstruction: NOT ESTABLISHED')
print('TRUTH_PACKET_JSON=' + json.dumps(results, sort_keys=True))










"""

Results:





SIM17 / SIM14 ORDER-192 DUAL-MODULE AUDIT: ALL ASSERTIONS PASSED
SIM14 affine: |H|=192, |Z(H)|= 2
SIM17 signed: |H|=192, |Z(H)|= 1
Shared quotient |Q|= 24  acts faithfully on four Fano sectors (S4)
Common fixed vectors: [0, 4]
Common fixed characters: [0]
Vector orbits (nonzero): [[4], [1, 2, 3, 5, 6, 7]]
Dual character orbits (nonzero): [[1, 2, 3], [4, 5, 6, 7]]
Octavian support-rich planes: [[0, 1, 4, 5], [0, 2, 4, 6], [0, 3, 4, 7]]
Root counts in rich/plain quaternionic sectors: 24 / 8 (3+4 sectors)
Q-invariant pairing <t,x> = parity(t & x): PASS
Canonical binary alternating form on V+V*: PASS (NOT frozen real Omega)
Held-out SIM14 directions in SIM17 coordinate map: {'p': 4, 'h': 2, 's_omega': 1}
Parent root-system selection and frozen OPS reconstruction: NOT ESTABLISHED
TRUTH_PACKET_JSON={"arithmetic_plane_split": [3, 4], "associative_plane_count": 7, "canonical_binary_symplectic_6D_Q_invariant": true, "center_SIM14_affine_order": 2, "center_SIM17_signed_order": 1, "common_fixed_characters_mod2": [0], "common_fixed_vectors_mod2": [0, 4], "count_octavian_lattice_preserving_signed_automorphisms": 192, "count_signed_basis_automorphisms": 1344, "dual_nonzero_character_orbits": [[1, 2, 3], [4, 5, 6, 7]], "dual_pairing_Q_invariant": true, "groups_isomorphic": false, "held_out_OPS_directions_in_SIM17_coordinate_map": {"h": 2, "p": 4, "s_omega": 1}, "independent_frozen_OPS_reconstruction": false, "normal_modules_Q_equivariantly_isomorphic": false, "octavian_norm_one_unit_counts_in_fano_planes": {"(0, 1, 2, 3)": 8, "(0, 1, 4, 5)": 24, "(0, 1, 6, 7)": 8, "(0, 2, 4, 6)": 24, "(0, 2, 5, 7)": 8, "(0, 3, 4, 7)": 24, "(0, 3, 5, 6)": 8}, "planes_with_half_integral_octavian_support": [[0, 1, 4, 5], [0, 2, 4, 6], [0, 3, 4, 7]], "planes_without_half_integral_octavian_support": [[0, 1, 2, 3], [0, 1, 6, 7], [0, 2, 5, 7], [0, 3, 5, 6]], "primal_nonzero_direction_orbits": [[4], [1, 2, 3, 5, 6, 7]], "quotient_is_S4_via_four_plane_action": true, "quotient_order": 24, "selection_of_octavian_order_from_parent": "NOT_TESTED", "source": "SIM14.8 / SIM17.4 independently reconstructed finite inputs"}
"""
