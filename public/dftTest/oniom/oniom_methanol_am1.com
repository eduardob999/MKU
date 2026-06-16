%chk=oniom_methanol_am1.chk
%mem=2GB
%nprocshared=28
#p ONIOM(HF/6-31G(d):AM1) Opt

ONIOM methanol (QM: HF/6-31G(d), LL: AM1)

0 1
C       0.000000    0.000000    0.000000 L
H       0.000000    0.000000    1.089000 L
H       1.026719    0.000000   -0.363000 L
H      -0.513360   -0.889165   -0.363000 L
O       0.000000    0.001000   -1.430000 H
H       0.000000    0.001000   -2.430000 H


