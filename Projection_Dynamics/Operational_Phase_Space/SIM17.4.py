
#!/usr/bin/env python3
"""SIM17.4 AMENDED — exact parent projection, arithmetic and h obstruction.

Gates A-G. All mathematical certificates use Python/NumPy integer arithmetic.
No dynamics or OPS-target information enters Gates A-F.
The octavian support code is a DECLARED input inherited from SIM17.1,
not independently selected. Exact coordinate carrier census is restricted.
"""
import itertools as it
import json
from collections import Counter
from math import comb
import numpy as np


def check(msg, assertion):
    if not bool(assertion):
        raise AssertionError("FAIL | " + msg)
    print("PASS | " + msg)


def cd(i, j, n=16):
    if n == 1:
        return 1, 0
    k = n//2
    if i < k and j < k: return cd(i,j,k)
    if i < k and j >= k:
        s,t = cd(j-k,i,k); return s,t+k
    if i >= k and j < k:
        s,t = cd(i-k,j,k); return s*(1 if j==0 else -1),t+k
    s,t = cd(j-k,i-k,k)
    return -s*(1 if j-k==0 else -1),t

SG=np.zeros((16,16),dtype=np.int64)
IX=np.zeros((16,16),dtype=np.int64)
for i,j in it.product(range(16),repeat=2): SG[i,j],IX[i,j]=cd(i,j)
I16=np.eye(16,dtype=np.int64)
I8=np.eye(8,dtype=np.int64)
W=tuple(range(8)); U=tuple(range(8,16))

def mult(x,y):
    z=np.zeros(16,dtype=np.int64)
    for i in np.flatnonzero(x):
        for j in np.flatnonzero(y):
            z[int(IX[i,j])] += int(x[i])*int(y[j])*int(SG[i,j])
    return z

def left(i):
    a=np.zeros((16,16),dtype=np.int64)
    for j in range(16): a[int(IX[i,j]),j]=SG[i,j]
    return a
J=[left(i) for i in range(16)]

def oprod(x,y):
    return mult(
        np.r_[x,np.zeros(8,dtype=np.int64)],
        np.r_[y,np.zeros(8,dtype=np.int64)]
    )[:8]

def conj8(x):
    y=x.copy(); y[1:]*=-1; return y

def output_rank(rows,cols,out):
    return len({
        int(IX[i,j])
        for i,j in it.product(rows,cols)
        if int(IX[i,j]) in out
    })

def exact_det(M):
    """Bareiss fraction-free determinant, arbitrary Python integers."""
    a=[[int(x) for x in row] for row in M]
    n=len(a); sign=1; prev=1
    if n==1:return a[0][0]
    for k in range(n-1):
        if a[k][k]==0:
            swap=next(
                (j for j in range(k+1,n) if a[j][k]),
                None
            )
            if swap is None:return 0
            a[k],a[swap]=a[swap],a[k]; sign=-sign
        piv=a[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                num=a[i][j]*piv-a[i][k]*a[k][j]
                if num%prev:
                    raise ArithmeticError('nonexact Bareiss division')
                a[i][j]=num//prev
        for i in range(k+1,n):a[i][k]=0
        prev=piv
    return sign*a[-1][-1]


print('='*67)
print('SIM17.4 — AMENDED EXACT PARENT-PROJECTION VALIDATOR')
print('='*67)


# ============================================================
# A: EXACT PARENT, CLIFFORD VOLUME, SPECTRAL PROJECTIONS
# ============================================================

print('\n--- GATE A: CAYLEY-DICKSON AND CLIFFORD VOLUME ---')

check(
    'A full signed XOR algebra, identity, imaginary squares',
    all(
        int(IX[i,j])==i^j
        for i,j in it.product(range(16),repeat=2)
    )
    and all(SG[0,i]==SG[i,0]==1 for i in range(16))
    and all(SG[i,i]==-1 for i in range(1,16))
)

check(
    'A all 15 imaginary left operators are orthogonal complex structures',
    all(
        np.array_equal(J[i]@J[i],-I16)
        and np.array_equal(J[i].T,-J[i])
        for i in range(1,16)
    )
)

check(
    'A seven original generators anticommute throughout the parent',
    all(
        not np.any(J[i]@J[j]+J[j]@J[i])
        for i,j in it.combinations(range(1,8),2)
    )
)

Gamma7=I16.copy()
Gamma8=I16.copy()

for i in range(1,8):
    Gamma7=Gamma7@J[i]

for i in range(8,16):
    Gamma8=Gamma8@J[i]

check(
    'A Gamma7 = -Gamma8 = diag(-I8,+I8)',
    np.array_equal(Gamma7,np.diag([-1]*8+[1]*8))
    and np.array_equal(Gamma8,-Gamma7)
)

P=(I16+Gamma8)//2
K=I16-P

check(
    'A two complementary rank-eight spectral projectors',
    np.array_equal(P@P,P)
    and np.array_equal(K@K,K)
    and not np.any(P@K)
    and int(np.trace(P))==8
    and int(np.trace(K))==8
)


# ============================================================
# B: COORDINATE CENSUS, MOVING AUTOMORPHISM, STABILIZER
# ============================================================

print('\n--- GATE B: COORDINATE LIMIT AND MOVING PARENT AUTOMORPHISM ---')

closed=[]
ambient=[]

for tail in it.combinations(range(1,16),7):
    S=(0,)+tail
    if all(
        int(IX[i,j]) in S
        for i,j in it.product(S,repeat=2)
    ):
        closed.append(S)
        ambient.append(
            sum(
                not np.any(J[i]@J[j]+J[j]@J[i])
                for i,j in it.combinations(S[1:],2)
            )
        )

check(
    'B coordinate census 6435 candidates, 15 closed, unique 21/21 coordinate carrier',
    comb(15,7)==6435
    and len(closed)==15
    and ambient.count(21)==1
    and closed[ambient.index(21)]==W
)

# Psi matrix is M/2, M=A+sqrt(3)B with integer A,B.
# Psi(e0)=e0; Psi(e8)=e8.
#
# Psi(e_i)=(-e_i-sqrt(3)e_{i+8})/2
# Psi(e_{i+8})=(sqrt(3)e_i-e_{i+8})/2
# for i=1..7.

A=np.zeros((16,16),dtype=np.int64)
B=np.zeros((16,16),dtype=np.int64)

A[0,0]=2
A[8,8]=2

for i in range(1,8):
    A[i,i]=-1
    B[i+8,i]=-1
    A[i+8,i+8]=-1
    B[i,i+8]=1

def pair_matmul(X,Y):
    """Multiply numerator pairs representing X+sqrt(3)Y."""
    (a,b),(c,d)=X,Y
    return a@c+3*(b@d),a@d+b@c

AA,BB=pair_matmul((A,B),(A,B))
AAA,BBB=pair_matmul((AA,BB),(A,B))

check(
    'B exact Q(sqrt3) automorphism has order three',
    np.array_equal(AAA,8*I16)
    and not np.any(BBB)
)

# Product of Psi(e_i) and Psi(e_j), numerator /2 each.
# Compare to Psi(e_i e_j), denominator /2.

for i,j in it.product(range(16),repeat=2):
    ai,bi=A[:,i],B[:,i]
    aj,bj=A[:,j],B[:,j]

    r=np.zeros(16,dtype=np.int64)
    s=np.zeros(16,dtype=np.int64)

    for k in range(16):
        for l in range(16):
            t=int(IX[k,l])
            f=int(SG[k,l])
            r[t]+=f*(
                int(ai[k])*int(aj[l])
                +3*int(bi[k])*int(bj[l])
            )
            s[t]+=f*(
                int(ai[k])*int(bj[l])
                +int(bi[k])*int(aj[l])
            )

    t=int(IX[i,j])
    f=int(SG[i,j])

    if not (
        np.array_equal(r,2*f*A[:,t])
        and np.array_equal(s,2*f*B[:,t])
    ):
        raise AssertionError(
            'FAIL | B Psi multiplication at '+str((i,j))
        )

check(
    'B Psi preserves all 256 parent multiplication products',
    True
)

check(
    'B Psi moves P and its carrier is unital and non-coordinate',
    np.any(A@P-P@A)
    or np.any(B@P-P@B)
)

# Check conjugate P'=Psi P Psi^-1
# using Psi^-1=Psi^2 and exact Q(sqrt3) arithmetic.
#
# P' numerator /8.

C_A,C_B=pair_matmul((A@P,B@P),(AA,BB))
T_A,T_B=pair_matmul((C_A,C_B),(C_A,C_B))

check(
    'B transported P has exact idempotence, unit and rank',
    np.array_equal(T_A,8*C_A)
    and np.array_equal(T_B,8*C_B)
    and int(np.trace(C_A))==64
    and int(np.trace(C_B))==0
    and np.array_equal(
        C_A[:,0],
        np.array([8]+[0]*15,dtype=np.int64)
    )
    and not (
        np.array_equal(C_A,8*P)
        and not np.any(C_B)
    )
)

print(
    'B | non-coordinate parent-conjugate rank-eight projector certified'
)


# ============================================================
# C: EXACT BILINEAR BLOCKS AND DEFECT IDENTITY
# ============================================================

print('\n--- GATE C: BILINEAR TENSOR BLOCKS ---')

blocks={
    name:output_rank(a,b,set(W))
    for name,a,b in [
        ('WW',W,W),
        ('WU',W,U),
        ('UW',U,W),
        ('UU',U,U)
    ]
}

check(
    'C full bilinear block profile (8,0,0,8)',
    blocks=={
        'WW':8,
        'WU':0,
        'UW':0,
        'UU':8
    }
)

check(
    'C exact tensor-kernel dimensions',
    {k:64-r for k,r in blocks.items()}
    =={
        'WW':56,
        'WU':64,
        'UW':64,
        'UU':56
    }
)

for i,j in it.product(range(16),repeat=2):
    x=I16[:,i]
    y=I16[:,j]

    defect=(
        P@mult(x,y)
        -P@mult(P@x,P@y)
    )

    if not np.array_equal(
        defect,
        P@mult(K@x,K@y)
    ):
        raise AssertionError(
            'FAIL | C projection defect at '+str((i,j))
        )

for i,j in it.product(range(8),repeat=2):
    x=I16[:,8+i]
    y=I16[:,8+j]

    if not np.array_equal(
        mult(x,y)[:8],
        -oprod(conj8(I8[:,j]),I8[:,i])
    ):
        raise AssertionError(
            'FAIL | C UU CD identity'
        )

check(
    'C defect exactly equals complementary UU product on all 256 basis pairs',
    True
)

check(
    'C UU product = -conj(d)b on all 64 complementary pairs',
    True
)


# ============================================================
# D: CONSTRUCT FOUR-PLANE PROJECTORS FROM ASSOCIATOR DEFECT
# ============================================================

print('\n--- GATE D: DEFECT-DERIVED QUATERNIONIC PROJECTORS ---')

L=[J[i][:8,:8] for i in range(8)]

projectors={}
qdata=[]

for i,j in it.combinations(range(1,8),2):
    c=i^j
    line=tuple(sorted((0,i,j,c)))

    if line in projectors:
        continue

    D=int(SG[i,j])*L[c]-L[i]@L[j]
    gram=D.T@D

    if np.any(gram%4):
        raise AssertionError(
            'FAIL | D nonintegral defect projector'
        )

    # The projector is DERIVED from the multiplication defect.
    # No coordinate mask is used to construct it.

    Q=I8-gram//4

    # Independent acceptance comparison.

    target=np.zeros((8,8),dtype=np.int64)
    for k in line:
        target[k,k]=1

    if not (
        np.array_equal(Q,target)
        and np.array_equal(Q@Q,Q)
        and np.array_equal(Q.T,Q)
        and np.trace(Q)==4
    ):
        raise AssertionError(
            'FAIL | D derivation of Q '+str(line)
        )

    # Associativity of the image.

    for a,b,c0 in it.product(line,repeat=3):
        u=oprod(
            oprod(I8[:,a],I8[:,b]),
            I8[:,c0]
        )
        v=oprod(
            I8[:,a],
            oprod(I8[:,b],I8[:,c0])
        )

        if not np.array_equal(u,v):
            raise AssertionError(
                'FAIL | D four-plane nonassoc'
            )

    R=np.zeros((16,16),dtype=np.int64)
    R[:8,:8]=Q

    ranks={
        nm:output_rank(x,y,set(line))
        for nm,x,y in [
            ('WW',W,W),
            ('WU',W,U),
            ('UW',U,W),
            ('UU',U,U)
        ]
    }

    if ranks!={'WW':4,'WU':0,'UW':0,'UU':4}:
        raise AssertionError(
            'FAIL | D composed bilinear ranks'
        )

    projectors[line]=Q
    qdata.append({
        'line':list(line),
        'projector_rank':4,
        'composed_bilinear_blocks':ranks
    })

check(
    'D seven projectors derived FROM defects, associative and rank four',
    len(projectors)==7
)

check(
    'D all composed bilinear maps have 4,0,0,4 block ranks',
    True
)


# ============================================================
# E: DECLARED OCTAVIAN LATTICE, EXACT BASIS, INCLUSION THEOREM
# ============================================================

print('\n--- GATE E: INTEGRAL ARITHMETIC AND NECESSITY CONTROL ---')

# This coordinate conversion is inherited from SIM17.1.
# Its original input presentation uses SIM14 indexing.
# It is supplied arithmetic data, NOT selected by the
# parent multiplication and NOT used to derive frozen OPS.

CONVERT=(0,4,1,5,7,3,6,2)

OLD_SUPPORTS=(
    (0,1,2,3),
    (0,1,4,5),
    (0,1,6,7),
    (0,2,4,7),
    (0,2,5,6),
    (0,3,4,6),
    (0,3,5,7),
    (1,2,4,6),
    (1,2,5,7),
    (1,3,4,7),
    (1,3,5,6),
    (2,3,4,5),
    (2,3,6,7),
    (4,5,6,7)
)

SUPPORTS=tuple(
    tuple(sorted(CONVERT[i] for i in S))
    for S in OLD_SUPPORTS
)

CODE=(
    {0,255}
    |{
        sum(1<<i for i in S)
        for S in SUPPORTS
    }
)

check(
    'E code has 16 doubly-even words and XOR closure',
    len(CODE)==16
    and Counter(x.bit_count() for x in CODE)
        =={0:1,4:14,8:1}
    and all(
        a^b in CODE
        for a,b in it.product(CODE,repeat=2)
    )
)


def in_order2(x):
    return sum(
        1<<i
        for i,a in enumerate(x)
        if int(a)%2
    ) in CODE


def in_z2(x):
    return all(int(a)%2==0 for a in x)


def octmul2(x,y):
    z=np.zeros(8,dtype=np.int64)
    for i in range(8):
        for j in range(8):
            z[int(IX[i,j])]+=(
                int(x[i])
                *int(y[j])
                *int(SG[i,j])
            )
    return tuple(int(t) for t in z)


def conj2(x):
    return (x[0],)+tuple(-t for t in x[1:])


# Each vector represents 2*x.
# Product of two doubled vectors represents 4*(xy).
# Divide by 2 to obtain doubled coordinates for xy.

def product_doubled(x,y):
    raw=octmul2(x,y)
    if any(t%2 for t in raw):
        raise ArithmeticError(
            'half-integrality failure'
        )
    return tuple(t//2 for t in raw)


# ------------------------------------------------------------
# E1 — 240 exact octavian units
# ------------------------------------------------------------

roots=set()

for i in range(8):
    for v in (-2,2):
        x=[0]*8
        x[i]=v
        roots.add(tuple(x))

for support in SUPPORTS:
    for signs in it.product((-1,1),repeat=4):
        x=[0]*8
        for i,v in zip(support,signs):
            x[i]=v
        roots.add(tuple(x))

roots=sorted(roots)

check(
    'E all 240 octavian unit vectors explicitly generated',
    len(roots)==240
    and all(
        sum(t*t for t in r)==4
        for r in roots
    )
)

for x,y in it.product(roots,repeat=2):
    if not in_order2(product_doubled(x,y)):
        raise AssertionError(
            'FAIL | E root product outside lattice'
        )

check(
    'E exact 57,600 unit-product closures',
    True
)


# ------------------------------------------------------------
# E2 — Exact integral basis
# ------------------------------------------------------------

# L2=2Z^8+CODE, representing 2*O.
# Four independent binary codewords plus 2e_i
# generate the lattice.
#
# Find an 8-vector basis with determinant magnitude 16.

code_basis=[]
subspace={0}

for c in sorted(CODE):
    if c not in subspace:
        code_basis.append(c)
        subspace={
            x^y
            for x in subspace
            for y in (0,c)
        }

check(
    'E binary code generated by four independent words',
    len(code_basis)==4
    and subspace==CODE
)

std=[
    tuple(
        2 if j==i else 0
        for j in range(8)
    )
    for i in range(8)
]

code_vecs=[
    tuple(
        (c>>i)&1
        for i in range(8)
    )
    for c in code_basis
]

gens=std+code_vecs
lat_basis=None

for combo in it.combinations(range(12),8):
    cols=[gens[i] for i in combo]
    det=exact_det(list(zip(*cols)))

    if abs(det)==16:
        lat_basis=cols
        break

check(
    'E explicit exact lattice basis covolume 16 in doubled coordinates',
    lat_basis is not None
)

for a,b in it.product(lat_basis,repeat=2):
    if not in_order2(product_doubled(a,b)):
        raise AssertionError(
            'FAIL | E basis product outside integral order'
        )

check(
    'E basis closure extends bilinearly to entire octavian lattice',
    True
)

check(
    'E lattice contains 1 and is conjugation stable',
    in_order2(std[0])
    and all(
        in_order2(conj2(b))
        for b in lat_basis
    )
)


# ------------------------------------------------------------
# E3 — Arithmetic inclusion theorem and negative control
# ------------------------------------------------------------

# For a unital, conjugation-stable,
# multiplication-closed octonionic lattice B:
#
#    conjugate(B)*B = B
#
# because 1 belongs to B and products are in B.
#
# Thus:
#
#    conjugate(B)*B subset A  iff  B subset A.
#
# The four arithmetic outcomes below certify lattice
# containment, not new parent-specific necessity.

O_GEN=lat_basis
Z_GEN=std

lattice_tests={}

for aname,Amem in [
    ('Z8',in_z2),
    ('O',in_order2)
]:
    for bname,Bgens in [
        ('Z8',Z_GEN),
        ('O',O_GEN)
    ]:
        subset=all(
            Amem(x)
            for x in Bgens
        )

        products=all(
            Amem(
                product_doubled(conj2(x),y)
            )
            for x,y in it.product(Bgens,repeat=2)
        )

        if subset!=products:
            raise AssertionError(
                'FAIL | E inclusion equivalence'
            )

        lattice_tests[
            'A='+aname+'_B='+bname
        ]=products

check(
    'E four arithmetic controls reduce EXACTLY to lattice inclusion',
    lattice_tests=={
        'A=Z8_B=Z8':True,
        'A=Z8_B=O':False,
        'A=O_B=Z8':True,
        'A=O_B=O':True
    }
)

check(
    'E octonion-only arithmetic control reproduces parent UU membership',
    all(
        in_order2(
            product_doubled(conj2(x),y)
        )
        for x,y in it.product(O_GEN,repeat=2)
    )
)

# Psi has irrational coefficients, hence it does not
# preserve the supplied rational product lattice O+O.

check(
    'E projector-moving Psi fails to preserve rational O+O lattice',
    np.any(B)
)

print('E | lattice pair results:',lattice_tests)


# ============================================================
# F: TWO SYMMETRY REGIMES, TRANSLATION ORBITS, PARENT PROVENANCE
# ============================================================

print('\n--- GATE F: PARENT AND ARITHMETIC STABILIZER ORBITS ---')

autos=[]

for a,b,c in it.product(range(1,8),repeat=3):
    if b==a or c in (a,b,a^b):
        continue

    perm=tuple(
        (a if i&1 else 0)
        ^(b if i&2 else 0)
        ^(c if i&4 else 0)
        for i in range(8)
    )

    for mask in range(128):
        signs=(1,)+tuple(
            -1 if mask & (1<<(i-1)) else 1
            for i in range(1,8)
        )

        if all(
            int(SG[i,j])*signs[i^j]
            ==
            signs[i]*signs[j]
            *int(SG[perm[i],perm[j]])
            for i,j in it.product(range(8),repeat=2)
        ):
            autos.append((perm,signs))

check(
    'F exactly 1344 signed octonion basis automorphisms',
    len(autos)==1344
)


# ------------------------------------------------------------
# F2 — Certify 16D lifts
# ------------------------------------------------------------

for perm,signs in autos:

    def pf(i):
        return perm[i&7]+(i&8)

    def sf(i):
        return signs[i&7]

    if not all(
        int(SG[i,j])*sf(i^j)
        ==
        sf(i)*sf(j)
        *int(SG[pf(i),pf(j)])
        and int(IX[pf(i),pf(j)])==pf(i^j)
        for i,j in it.product(range(16),repeat=2)
    ):
        raise AssertionError(
            'FAIL | F parent lift'
        )

check(
    'F all 1344 lift to 16D multiplication automorphisms fixing P',
    True
)


# ------------------------------------------------------------
# F3 — Arithmetic-lattice-preserving subgroup
# ------------------------------------------------------------

support_set={
    frozenset(S)
    for S in SUPPORTS
}

preserving=[
    (perm,signs)
    for perm,signs in autos
    if {
        frozenset(perm[i] for i in S)
        for S in SUPPORTS
    }==support_set
]

check(
    'F precisely 192 certified signed-basis symmetries preserve octavian supports',
    len(preserving)==192
)


def orbit_partition(group):
    pending=set(range(1,8))
    result=[]

    while pending:
        i=min(pending)
        orb={
            perm[i]
            for perm,_ in group
        }

        assert orb<=pending
        pending-=orb
        result.append(sorted(orb))

    return sorted(
        result,
        key=lambda v:(len(v),v)
    )


base_orbits=orbit_partition(autos)
lattice_orbits=orbit_partition(preserving)

check(
    'F multiplication-only symmetry gives one 7-direction orbit',
    base_orbits==[[1,2,3,4,5,6,7]]
)

check(
    'F supplied arithmetic reduces certified orbits to 1+6',
    lattice_orbits==[
        [4],
        [1,2,3,5,6,7]
    ]
)


# ------------------------------------------------------------
# F4 — Seven unsigned translation candidates
# ------------------------------------------------------------

translations={
    a:tuple(x^a for x in range(8))
    for a in range(1,8)
}

check(
    'F exactly seven fixed-point-free quotient translations',
    len(set(translations.values()))==7
    and all(
        all(
            t[t[x]]==x and t[x]!=x
            for x in range(8)
        )
        for t in translations.values()
    )
)

print('F | parent-only orbits:',base_orbits)
print(
    'F | parent+supplied-arithmetic restricted orbits:',
    lattice_orbits
)


# ============================================================
# G: HELD-OUT SIM14 OPS ACCEPTANCE; NO UPSTREAM FITTING
# ============================================================

print('\n--- GATE G: HELD-OUT SIM14–16 OPS ACCEPTANCE ---')

# The frozen OPS target first enters here.
# Nothing below was used to generate Gates A-F candidates.

FROZEN_EDGES={
    (0,1),
    (0,2),
    (1,3),
    (2,4),
    (3,5),
    (4,6),
    (5,7),
    (6,7)
}

FROZEN_P=(1,0,3,2,5,4,7,6)
FROZEN_H=(7,6,5,4,3,2,1,0)
FROZEN_OMEGA_PAIRS=(
    (0,2),
    (1,3),
    (4,6),
    (5,7)
)

raw_h=[
    a
    for a,t in translations.items()
    if t==FROZEN_H
]

check(
    'G raw frozen h is one member of the unselected family',
    raw_h==[7]
)

check(
    'G lattice-stabilizer singleton is NOT raw frozen h',
    4 not in raw_h
)

print(
    'G | raw h match:',
    raw_h,
    '; symmetry singleton:',
    lattice_orbits[0]
)

print(
    'G | frozen p, Omega, observation B, J_t and labeled C8 NOT DERIVED'
)


# ============================================================
# MACHINE-READABLE TRUTH PACKET
# ============================================================

packet={
    'sim':'17.4',

    'amendment':
        'PARENT_AUTOMORPHISM_ARITHMETIC_STABILIZER',

    'overall':
        'PASS_WITH_CERTIFIED_NONUNIQUENESS_AND_SELECTION_LIMITS',

    'A_clifford_volumes':
        'Gamma7=-Gamma8, projectors rank8+8',

    'B_closed_coordinate_eight_planes':
        len(closed),

    'B_unique_coordinate_full_clifford':
        closed[ambient.index(21)],

    'B_parent_noncoordinate_order3_automorphism':
        True,

    'B_projector_moves_under_parent_automorphism':
        True,

    'B_full_parent_automorphism_group_enumerated':
        False,

    'C_bilinear_block_ranks':
        blocks,

    'C_tensor_kernels':{
        k:64-v
        for k,v in blocks.items()
    },

    'C_defect_equals_complement_UU':
        True,

    'D_quaternionic_derived_projectors':
        qdata,

    'E_declared_octavian_root_count':
        len(roots),

    'E_exact_unit_products_checked':
        len(roots)**2,

    'E_doubled_lattice_basis':
        lat_basis,

    'E_doubled_lattice_basis_determinant_abs':
        16,

    'E_arithmetic_controls':
        lattice_tests,

    'E_arithmetic_is_lattice_inclusion_control':
        True,

    'E_parent_specific_arithmetic_necessity_established':
        False,

    'E_projector_moving_Psi_preserves_supplied_rational_lattice':
        False,

    'F_signed_basis_parent_symmetry_subgroup':
        len(autos),

    'F_lattice_preserving_signed_basis_subgroup':
        len(preserving),

    'F_parent_only_direction_orbits':
        base_orbits,

    'F_arithmetic_stabilizer_direction_orbits':
        lattice_orbits,

    'G_frozen_h_raw_index_matches':
        raw_h,

    'G_frozen_h_independently_selected':
        False,

    'G_frozen_OPS_independently_recovered':
        False,

    'provenance':{
        'parent':
            'Fixed canonical 16D Cayley-Dickson basis and Euclidean metric',

        'spectral':
            'Chosen Clifford generator family; projector not parent-intrinsic',

        'coordinate_census':
            '6435 unital coordinate 8-planes only',

        'arithmetic':
            'Octavian supports supplied from SIM17.1; original presentation uses SIM14 indexing',

        'symmetry':
            '1344 signed-basis lifts, 192 preserving supplied lattice; not full parent/lattice group',

        'half_turn':
            'Seven unsigned XOR translations; raw frozen h first checked in Gate G',

        'dynamics':
            'NONE; purely algebraic admissibility'
    }
}


# ============================================================
# SAVE / PRINT VALIDATION RESULTS
# ============================================================

try:
    filename='SIM17.4_amended_truth_packet.json'

    with open(filename,'w',encoding='utf-8') as fp:
        json.dump(packet,fp,indent=2)

    print('Truth packet saved:',filename)

except OSError as err:
    print(
        'Truth packet file write unavailable:',
        err
    )


print(
    'TRUTH_PACKET_SUMMARY='
    +json.dumps(
        {
            k:v
            for k,v in packet.items()
            if k!='D_quaternionic_derived_projectors'
        },
        sort_keys=True
    )
)


# ============================================================
# FINAL VERDICT
# ============================================================

print('\n'+'='*67)
print('SIM17.4 AMENDED VERDICT')

print(
    'PASS | Clifford projectors, bilinear UU defect and 4D defect projectors'
)

print(
    'PASS | Explicit order-3 automorphism proves P nonunique in parent'
)

print(
    'PASS | Lattice inclusion theorem and exact octavian basis controls'
)

print(
    'PASS | 1344 -> 192 arithmetic symmetry reduction, direction orbit 1+6'
)

print(
    'OPEN | Intrinsic projector / half-turn / frozen OPS selection'
)

print(
    'NO DYNAMICS | ISP x LCO not invoked'
)

print('END SIM17.4 AMENDED')







"""




RESULTS:





===================================================================
SIM17.4 — AMENDED EXACT PARENT-PROJECTION VALIDATOR
===================================================================

--- GATE A: CAYLEY-DICKSON AND CLIFFORD VOLUME ---PASS | A full signed XOR algebra, identity, imaginary squares
PASS | A all 15 imaginary left operators are orthogonal complex structures
PASS | A seven original generators anticommute throughout the parent
PASS | A Gamma7 = -Gamma8 = diag(-I8,+I8)
PASS | A two complementary rank-eight spectral projectors

--- GATE B: COORDINATE LIMIT AND MOVING PARENT AUTOMORPHISM ---PASS | B coordinate census 6435 candidates, 15 closed, unique 21/21 coordinate carrier
PASS | B exact Q(sqrt3) automorphism has order three
PASS | B Psi preserves all 256 parent multiplication products
PASS | B Psi moves P and its carrier is unital and non-coordinate
PASS | B transported P has exact idempotence, unit and rank
B | non-coordinate parent-conjugate rank-eight projector certified

--- GATE C: BILINEAR TENSOR BLOCKS ---PASS | C full bilinear block profile (8,0,0,8)
PASS | C exact tensor-kernel dimensions
PASS | C defect exactly equals complementary UU product on all 256 basis pairs
PASS | C UU product = -conj(d)b on all 64 complementary pairs

--- GATE D: DEFECT-DERIVED QUATERNIONIC PROJECTORS ---PASS | D seven projectors derived FROM defects, associative and rank four
PASS | D all composed bilinear maps have 4,0,0,4 block ranks

--- GATE E: INTEGRAL ARITHMETIC AND NECESSITY CONTROL ---PASS | E code has 16 doubly-even words and XOR closure
PASS | E all 240 octavian unit vectors explicitly generated
PASS | E exact 57,600 unit-product closures
PASS | E binary code generated by four independent words
PASS | E explicit exact lattice basis covolume 16 in doubled coordinates
PASS | E basis closure extends bilinearly to entire octavian lattice
PASS | E lattice contains 1 and is conjugation stable
PASS | E four arithmetic controls reduce EXACTLY to lattice inclusion
PASS | E octonion-only arithmetic control reproduces parent UU membership
PASS | E projector-moving Psi fails to preserve rational O+O lattice
E | lattice pair results: {'A=Z8_B=Z8': True, 'A=Z8_B=O': False, 'A=O_B=Z8': True, 'A=O_B=O': True}

--- GATE F: PARENT AND ARITHMETIC STABILIZER ORBITS ---PASS | F exactly 1344 signed octonion basis automorphisms
PASS | F all 1344 lift to 16D multiplication automorphisms fixing P
PASS | F precisely 192 certified signed-basis symmetries preserve octavian supports
PASS | F multiplication-only symmetry gives one 7-direction orbit
PASS | F supplied arithmetic reduces certified orbits to 1+6
PASS | F exactly seven fixed-point-free quotient translations
F | parent-only orbits: [[1, 2, 3, 4, 5, 6, 7]]
F | parent+supplied-arithmetic restricted orbits: [[4], [1, 2, 3, 5, 6, 7]]

--- GATE G: HELD-OUT SIM14–16 OPS ACCEPTANCE ---PASS | G raw frozen h is one member of the unselected family
PASS | G lattice-stabilizer singleton is NOT raw frozen h
G | raw h match: [7] ; symmetry singleton: [4]
G | frozen p, Omega, observation B, J_t and labeled C8 NOT DERIVED
Truth packet saved: SIM17.4_amended_truth_packet.json
TRUTH_PACKET_SUMMARY={"A_clifford_volumes": "Gamma7=-Gamma8, projectors rank8+8", "B_closed_coordinate_eight_planes": 15, "B_full_parent_automorphism_group_enumerated": false, "B_parent_noncoordinate_order3_automorphism": true, "B_projector_moves_under_parent_automorphism": true, "B_unique_coordinate_full_clifford": [0, 1, 2, 3, 4, 5, 6, 7], "C_bilinear_block_ranks": {"UU": 8, "UW": 0, "WU": 0, "WW": 8}, "C_defect_equals_complement_UU": true, "C_tensor_kernels": {"UU": 56, "UW": 64, "WU": 64, "WW": 56}, "E_arithmetic_controls": {"A=O_B=O": true, "A=O_B=Z8": true, "A=Z8_B=O": false, "A=Z8_B=Z8": true}, "E_arithmetic_is_lattice_inclusion_control": true, "E_declared_octavian_root_count": 240, "E_doubled_lattice_basis": [[2, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 1]], "E_doubled_lattice_basis_determinant_abs": 16, "E_exact_unit_products_checked": 57600, "E_parent_specific_arithmetic_necessity_established": false, "E_projector_moving_Psi_preserves_supplied_rational_lattice": false, "F_arithmetic_stabilizer_direction_orbits": [[4], [1, 2, 3, 5, 6, 7]], "F_lattice_preserving_signed_basis_subgroup": 192, "F_parent_only_direction_orbits": [[1, 2, 3, 4, 5, 6, 7]], "F_signed_basis_parent_symmetry_subgroup": 1344, "G_frozen_OPS_independently_recovered": false, "G_frozen_h_independently_selected": false, "G_frozen_h_raw_index_matches": [7], "amendment": "PARENT_AUTOMORPHISM_ARITHMETIC_STABILIZER", "overall": "PASS_WITH_CERTIFIED_NONUNIQUENESS_AND_SELECTION_LIMITS", "provenance": {"arithmetic": "Octavian supports supplied from SIM17.1; original presentation uses SIM14 indexing", "coordinate_census": "6435 unital coordinate 8-planes only", "dynamics": "NONE; purely algebraic admissibility", "half_turn": "Seven unsigned XOR translations; raw frozen h first checked in Gate G", "parent": "Fixed canonical 16D Cayley-Dickson basis and Euclidean metric", "spectral": "Chosen Clifford generator family; projector not parent-intrinsic", "symmetry": "1344 signed-basis lifts, 192 preserving supplied lattice; not full parent/lattice group"}, "sim": "17.4"}

===================================================================SIM17.4 AMENDED VERDICT
PASS | Clifford projectors, bilinear UU defect and 4D defect projectors
PASS | Explicit order-3 automorphism proves P nonunique in parent
PASS | Lattice inclusion theorem and exact octavian basis controls
PASS | 1344 -> 192 arithmetic symmetry reduction, direction orbit 1+6
OPEN | Intrinsic projector / half-turn / frozen OPS selection
NO DYNAMICS | ISP x LCO not invoked
END SIM17.4 AMENDED
