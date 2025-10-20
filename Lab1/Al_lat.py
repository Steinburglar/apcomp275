"""
script meant to be flexible up to problem 1 of lab 1. not flexible for vacancies or supercells
"""


from labutil.plugins.lammps import lammps_run, get_lammps_energy
from labutil.objects import Struc, Dir, ClassicalPotential, ase2struc
from ase.spacegroup import crystal
from ase.build import make_supercell
import numpy, os
import matplotlib.pyplot as plt
import argparse


def make_struc(alat, super = False, size =2,):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    unitcell = crystal(
        "Al", [(0, 0, 0)], spacegroup=225, cellpar=[alat, alat, alat, 90, 90, 90]
    )
    if super:
        multiplier = numpy.identity(3) * size
        ase_supercell = make_supercell(unitcell, multiplier)
        structure = Struc(ase2struc(ase_supercell))
    else:
        structure = Struc(ase2struc(unitcell))

    return structure


def compute_energy(alat, template, super = False, relax = False):
    """
    Make an input template and select potential and structure, and the path where to run
    """

    potpath = os.path.join(os.environ["LAMMPS_POTENTIALS"], "Al_zhou.eam.alloy")  #path to desired potential function
    if args.potential == 'eam':
        potential = ClassicalPotential(path=potpath, ptype="eam", element=["Al"])   #define potential object
    elif args.potential == 'lj':
        potential = None
    runpath = Dir(path=os.path.join(os.environ["WORKDIR"], run_dir, str(alat)))  #pass a directory object to variable runpath
    struc = make_struc(alat=alat, super=super)   #builds structure object based on lattice parameter
    output_file = lammps_run(
        struc=struc,
        runpath=runpath,
        potential=potential, 
        intemplate=template,
        inparam={},
    )
    energy, lattice = get_lammps_energy(outfile=output_file)   #labutils function. lattice is just lattice constant
    return energy, lattice


def lattice_scan():
    alat_list = numpy.linspace(3.9, 4.3, 6)
    energy_list = [
        compute_energy(alat=a, template=input_template, super = args.super, relax = False)[0] for a in alat_list
    ]
    size = 32 if args.super else 4
    cohesive_energy = [e/size for e in energy_list]  #energy per atom

    outtxt = f'Lattice constants: {alat_list} A, \n Cohesive energies: {cohesive_energy} eV'
    print(outtxt)
    with open(os.path.join(os.environ["WORKDIR"], run_dir, "scan_result.txt"), 'w') as f:
        f.write(outtxt)
    plt.plot(alat_list, energy_list)
    title = 'Lattice Scan Result,' +f' {args.potential.upper()}'
    if args.super:
        title += ', Supercell'
    plt.title(title)
    plt.savefig(os.path.join(os.environ["WORKDIR"], run_dir, "scan_result.png"))
    plt.show()

def lattice_relax():
    a = 4.05 #rough initial guess based on exp results
    size = 32 if args.super else 4
    energy, lattice = compute_energy(alat=a, template=input_template, super = args.super, relax = True)
    cohesive_energy = energy/size

    outtxt = f'Relaxed lattice constant: {lattice} A, Energy per atom: {cohesive_energy} eV'
    print(outtxt)
    with open(os.path.join(os.environ["WORKDIR"], run_dir, "relaxation_result.txt"), 'w') as f:
        f.write(outtxt)
    plt.plot(a, energy, 'o')
    title = 'Lattice Relaxation Result,' +f' {args.potential.upper()}'
    if args.super:
        title += ', Supercell'
    plt.title(title)
    plt.savefig(os.path.join(os.environ["WORKDIR"], run_dir, "relaxation_result.png"))
    plt.show()


if __name__ == "__main__":
    # put here the function that you actually want to run
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('potential', type=str, help='Potential to use: eam or lj')
    parser.add_argument('--relax', action='store_true', help='Perform lattice relaxation instead of scanning')
    parser.add_argument('--super', action='store_true', help='Use supercell')



    args = parser.parse_args()
    
    #check args to get appropriate input templates and run directories
    if args.potential == 'eam':
        if args.relax:
            input_file = "eam_relax.in"
            run_dir = "Lab1/EAM/relax"
        else:
            input_file = "eam.in"
            run_dir = "Lab1/EAM/scan"
    elif args.potential == 'lj':
        if args.relax:
            input_file = "lj_relax.in"
            run_dir = "Lab1/LJ/relax"   
        else:
            input_file = "lj.in"
            run_dir = "Lab1/LJ/scan"
    if args.super:
        run_dir += "/supercell"
    with open(input_file, 'r') as f:
        input_template = f.read()

    if args.relax:
        lattice_relax()
    else:
        lattice_scan()
