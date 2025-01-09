import numpy as np
from fibermodes import FiberFactory, Mode
from matplotlib import pyplot, pyplot as plt


factory = FiberFactory()
factory.addLayer(name="core", radius=[10e-6], index=1.50)
factory.addLayer(name="cladding",index=1.45)
for i, fiber in enumerate(factory):
    print(factory.layers[0].radius[i], fiber.neff(Mode("HE", 1, 1), 1.55e-6)) # Prints the Effective Refracive index of a GIVEN MODE at A Given Wavelenght
modee = Mode("HE", 1, 1)
field = fiber.field(modee, 1.55e-6, 10e-6, 101)
#print(ff)
#ha1 = ff.Hphi(0, 0)
#ha2 = ff.Hz(1, 2)
#ha3 = ff.Hr(1, 2)
#ah1 = ff.Ephi(0, 0)
#ah2 = ff.Ez(1, 2)
#ah3 = ff.Er(1, 2)

ah4 = field.Emod(0, 0) # absolute value of Ephi + Ez + Er
ah5 = field.Epol(0, 0)
fig, ax = plt.subplots(figsize=(20, 20))
polarization_angles = ah5
U = -np.cos(polarization_angles)
V = -np.sin(polarization_angles)
quiver = ax.quiver(U, V, color='black', scale=1, scale_units='xy', angles='xy')
plt.imshow(np.abs(ah4)**2, cmap='jet', interpolation='none')
mode_family = modee.family
nu = modee.nu
m = modee.m
plt.title(f'{mode_family} {nu}{m} Emod')
plt.colorbar()
plt.show()