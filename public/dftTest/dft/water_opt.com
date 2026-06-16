%chk=water_opt.chk
%mem=2GB
%nprocshared=28
#p hf/6-31G(d) opt

Simple water geometry optimization

0 1
O      0.000000   0.000000   0.000000
H      0.000000   0.757160   0.586260
H      0.000000  -0.757160   0.586260

