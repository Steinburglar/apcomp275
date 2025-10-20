#takes the values of energy for different lattice parameters, computes the bulk modulus by finding the second derivative of the energy with respect to the volume

#load in the csv as  a pandas dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('../lattice_parameter_convergence_results.csv')
volumes = df['Volume (Å^3)']
energies = df['Energy (eV)']
#find the index of the minimum energy
min_energy_index = energies.idxmin()
#find the corresponding volume
volume0= volumes[min_energy_index]

# Compute the first and second derivatives of energy with respect to volume
dE_dV = np.gradient(energies, volumes)
d2E_dV2 = np.gradient(dE_dV, volumes)
print(d2E_dV2)
# Compute the bulk modulus at the equilibrium volume
bulk_modulus =  volume0 * d2E_dV2[min_energy_index]
print(f'Bulk Modulus: {bulk_modulus} eV/Å^3')

# Convert bulk modulus to GPa (1 eV/Å^3 = 160.21766208 GPa)
bulk_modulus_GPa = bulk_modulus * 160.2176621
print(f'Bulk Modulus: {bulk_modulus_GPa} GPa')


#try by fitting a parabola to the curve, then taking second derrivative
coefficients = np.polyfit(volumes, energies, 2)
a = coefficients[0]  # Coefficient of V^2 term
# Bulk modulus B = V * d²E/dV² = V * 2a
bulk_modulus_fit = volume0 * 2 * a
bulk_modulus_fit_GPa = bulk_modulus_fit * 160.2176621
print(f'Bulk Modulus from fit: {bulk_modulus_fit} eV/Å^3')
print(f'Bulk Modulus from fit: {bulk_modulus_fit_GPa} GPa')

#plot  in three separate plots the energy vs volume curve with matplotlib, the first derrivative, and the second derrivative. label the axes and title
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(volumes, energies, marker='o')
plt.xlabel('Volume (Å^3)')
plt.ylabel('Energy (eV)')
plt.title('Energy vs Volume')
plt.axvline(x=volume0, color='r', linestyle='--', label='Equilibrium Volume')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(volumes, dE_dV, marker='o', color='g')
plt.xlabel('Volume (Å^3)')
plt.ylabel('dE/dV (eV/Å^3)')
plt.title('First Derivative of Energy vs Volume')
plt.axvline(x=volume0, color='r', linestyle='--', label='Equilibrium Volume')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(volumes, d2E_dV2, marker='o', color='m')
plt.xlabel('Volume (Å^3)')
plt.ylabel('d²E/dV² (eV/Å^6)')
plt.title('Second Derivative of Energy vs Volume')
plt.axvline(x=volume0, color='r', linestyle='--', label='Equilibrium Volume')
plt.legend()
plt.tight_layout()
plt.savefig('bulk_modulus_analysis.png')
plt.show()