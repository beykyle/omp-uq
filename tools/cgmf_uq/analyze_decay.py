from CGMFtk import histories as fh
h = fh.Histories("./histories.cgmf.0")
dj = h.getNeutronDeltaJ()
de = h.getNeutronEcm()
J0 = h.getJ()
Estar = h.getU()
A = h.getA()
Z = h.getA()

# plot E/E* vs J/J_0 for each neutron, avged over all A,Z

# plot E  vs J for each neutron, for single A,Z

