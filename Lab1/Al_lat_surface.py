"""
script for answering problem 3 of lab 1. computes surface energy of aluminum
"""

from labutil.plugins.lammps import lammps_run, get_lammps_energy
from labutil.objects import Struc, Dir, ClassicalPotential, ase2struc
from ase.spacegroup import crystal
from ase.build import make_supercell, fcc100, fcc110, fcc111
from ase.visualize import view
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

LJ_alat = 4.12252301377974 #Lattice constant in angstroms using LJ results
EAM_alat = 4.08165490486347 #Lattice constant in angstroms using EAM results

lj_cohesion = -2.6346046511628 #cohesive energy for LJ potential in eV
eam_cohesion = -3.579998659929 #cohesive energy for EAM potential in eV

def make_struc(alat, layers, vacuum, surface = '100'):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :param structure: "bulk" or "surface"
    :param surface: surface orientation, '100', '110', or '111'
    :param layers: number of layers in the slab
    :param vacuum: amount of vacuum in angstrom
    :return: structure object converted from ase, # of atoms in structure
    """
    
    if surface == '100':
        unitcell = fcc100('Al', size=(3, 3, layers), a=alat, vacuum=vacuum)
    elif surface == '110':
        unitcell = fcc110('Al', size=(3, 3, layers), a=alat, vacuum=vacuum)
    elif surface == '111':
        unitcell = fcc111('Al', size=(3, 3, layers), a=alat, vacuum=vacuum)
    else:
        raise ValueError("Surface must be '100', '110', or '111'")
    unitcell.periodic = (True, True, False)
    unitcell.edit()
    structure = Struc(ase2struc(unitcell))
    n = len(unitcell)
    return structure, n


def compute_surface_energy(alat, templates, layers, vacuum, surface ='100'):
    bulk_template, surface_template = templates
    Ebulk = cohesion
    Eslab, nslab = compute_energy(alat, surface_template, layers, vacuum, surface=surface)
    Esurf = (Eslab - nslab * Ebulk)  / (2 * ((3*alat)**2))  #surface energy in eV/angstrom^2
    return Esurf

def compute_energy(alat, template, layers, vacuum, surface = "100"):
    """
    Make an input template and select potential and structure, and the path where to run
    Make sure the struc matches the structure (boundary conditions) in the lammps input template
    :param alat: lattice parameter in angstrom
    :param template: lammps input template as a string
    :param struc: "bulk" or "surface"
    :param relax: if True, perform a lattice relaxation instead of single point energy calculation (implemented in the lammps input template)
    """
    potpath = os.path.join(os.environ["LAMMPS_POTENTIALS"], "Al_zhou.eam.alloy")  #path to desired potential function
    if args.potential == 'eam':
        potential = ClassicalPotential(path=potpath, ptype="eam", element=["Al"])   #define potential object
    elif args.potential == 'lj':
        potential = None
    runpath = Dir(path=os.path.join(os.environ["WORKDIR"], run_dir, f"{layers}_{vacuum}_{surface}"))  #pass a directory object to variable runpath
    struc, n_atoms = make_struc(alat, layers, vacuum, surface=surface)   #builds structure object based on lattice parameter
    output_file = lammps_run(
        struc=struc,
        runpath=runpath,
        potential=potential, 
        intemplate=template,
        inparam={},
    )
    energy, _ = get_lammps_energy(outfile=output_file)   #labutils function. lattice is just lattice constant
    return energy, n_atoms


def lattice_scan_surface(templates, surface = '100'):
    """
    Scan a range of lattice parameters and compute surface energies
    :param templates: duple of bulk and surface input template for lammps run
    :param surface: surface orientation, '100', '110', or '111'
    :param layers: number of layers in the slab
    :param vacuum: amount of vacuum in angstrom"""
    layers = np.arange(1, 20, 2)  #number of layers in the slab
    vacuum = np.arange(10, 11)  #amount of vacuum in angstrom
    L, V = np.meshgrid(layers, vacuum)
    energy_list = np.zeros_like(L, dtype=float)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            energy_list[i, j] = compute_surface_energy(
                true_alat, templates, L[i, j], V[i, j], surface=surface
            )
    outtxt = energy_list.__str__()
    with open(os.path.join(os.environ["WORKDIR"], run_dir, "scan_result.txt"), 'w') as f:
        f.write(outtxt)

    # Plot surface energy vs layers (L) for fixed vacuum (V=V[0])
    plt.figure()
    z = 0
    plt.plot(L[z, :], energy_list[z, :], marker='o')
    print(energy_list[z, :])
    plt.autoscale(enable=True, axis='y', tight=False)
    plt.xlabel('Number of Layers')
    plt.ylabel('Surface Energy (eV/Å²)')
    title_L = f'Surface {surface} Energy vs Layers, {args.potential.upper()}'
    if args.relax:
        title_L += ', Relaxed'
    plt.title(title_L)
    plt.savefig(os.path.join(os.environ["WORKDIR"], run_dir, "scan_result_layers.png"))

    # Plot surface energy vs vacuum (V) for fixed layers (L=L[:,0])
    plt.figure()
    plt.plot(V[:, 4], energy_list[:, 4], marker='o')
    plt.autoscale(enable=True, axis='y', tight=False)
    plt.xlabel('Vacuum (Å)')
    plt.ylabel('Surface Energy (eV/Å²)')
    title_V = f'Surface {surface} Energy vs Vacuum, {args.potential.upper()}'
    if args.relax:
        title_V += ', Relaxed'
    plt.title(title_V)
    plt.savefig(os.path.join(os.environ["WORKDIR"], run_dir, "scan_result_vacuum.png"))

    plt.show()


    


if __name__ == "__main__":
    # put here the function that you actually want to run
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('potential', type=str, help='Potential to use: eam or lj')
    parser.add_argument('--relax', action='store_true', help='Perform lattice relaxation instead of scanning')
    parser.add_argument('--surface', type=str, default='100', help='Surface orientation: 100, 110, or 111')



    args = parser.parse_args()
    
    #check args to get appropriate input templates and run directories
    #I know this code is sloppy but I didnt want to deal
    if args.potential == 'eam':
        cohesion = eam_cohesion
        true_alat = EAM_alat
        if args.relax:
            bulk_file = "eam_relax.in"
            surface_file = "eam_surface_relax.in"
            run_dir = "Lab1/Surface/EAM/relax"
        else:
            bulk_file = "eam.in"
            surface_file = "eam_surface.in"
            run_dir = "Lab1/Surface/EAM/scan"
    elif args.potential == 'lj':
        cohesion = lj_cohesion
        true_alat = LJ_alat
        if args.relax:
            bulk_file = "lj_relax.in"
            surface_file = "lj_surface_relax.in"
            run_dir = "Lab1/Surface/LJ/relax"
        else:
            bulk_file = "lj.in"
            surface_file = "lj_surface.in"
            run_dir = "Lab1/Surface/LJ/scan"
    with open(bulk_file, 'r') as f:
        bulk_template = f.read()
    with open(surface_file, 'r') as f:
        surface_template = f.read()

    lattice_scan_surface(templates=(bulk_template, surface_template), surface=args.surface)
