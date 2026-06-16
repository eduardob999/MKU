%chk=oniom_methanol.chk
%mem=2GB
%nprocshared=28
#p ONIOM(HF/6-31G(d):UFF) Freq Geom=Check Guess=Read

ONIOM methanol frequency from optimized checkpoint

0 1
