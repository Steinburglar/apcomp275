"""
script meant to be flexible up to problem 2 of lab 1. deals with vacancies and supercells
"""


from labutil.plugins.lammps import lammps_run, get_lammps_energy
from labutil.objects import Struc, Dir, ClassicalPotential, ase2struc
from ase.spacegroup import crystal
from ase.build import make_supercell
import numpy, os
import matplotlib.pyplot as plt
import argparse

LJ_alat = 4.12252301377974 #Lattice constant in angstroms using LJ results
EAM_alat = 4.08165490486347 #Lattice constant in angstroms using EAM results

lj_cohesion = -2.6346046511628 #cohesive energy for LJ potential in eV
eam_cohesion = -3.579998659929 #cohesive energy for EAM potential in eV


def make_struc(alat, size = 1):
    """
    Creates the crystal structure using ASE, and removes an atom
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    unitcell = crystal(
        "Al", [(0, 0, 0)], spacegroup=225, cellpar=[alat, alat, alat, 90, 90, 90]
    )
    multiplier = numpy.identity(3) * size
    ase_supercell = make_supercell(unitcell, multiplier)
    ase_supercell.pop(2)  #remove an atom to create vacancy
    structure = Struc(ase2struc(ase_supercell))
    return structure

def compute_vacancy_energy(alat, template, size = 1):
    n = size**3 * 4  #number of atoms in supercell
    Eperf = cohesion  #energy per atom in perfect crystal
    Evac = compute_energy(alat, template, size=size)[0]
    Evacancy = Evac - (n - 1) * Eperf
    return Evacancy


def compute_energy(alat, template, size = 1):
    """
    Make an input template and select potential and structure, and the path where to run
    """

    potpath = os.path.join(os.environ["LAMMPS_POTENTIALS"], "Al_zhou.eam.alloy")  #path to desired potential function
    if args.potential == 'eam':
        potential = ClassicalPotential(path=potpath, ptype="eam", element=["Al"])   #define potential object
    elif args.potential == 'lj':
        potential = None
    runpath = Dir(path=os.path.join(os.environ["WORKDIR"], run_dir, str(size)))  #pass a directory object to variable runpath
    struc = make_struc(alat=alat, size = size)   #builds structure object based on lattice parameter
    output_file = lammps_run(
        struc=struc,
        runpath=runpath,
        potential=potential, 
        intemplate=template,
        inparam={},
    )
    energy, lattice = get_lammps_energy(outfile=output_file)   #labutils function. lattice is just lattice constant
    return energy, lattice


def lattice_sizescan():
    a = true_alat
    size_list = numpy.arange(1, 15)
    energy_list = [
        compute_vacancy_energy(alat=true_alat, template=input_template, size = s) for s in size_list
    ]
    #generate a table to save
    outtxt = 'Size\tVacancy Energy (eV)\t Ratio\n'
    for s, e in zip(size_list, energy_list):
        outtxt += f'{s}\t{e}\t{-e/cohesion}\n'
    print(outtxt)
    with open(os.path.join(os.environ["WORKDIR"], run_dir, "scan_result.txt"), 'w') as f:
        f.write(outtxt)
    plt.plot(size_list, energy_list)
    title = 'Lattice Vacancy Scan Result,' +f' {args.potential.upper()}'
    if args.relax:
        title += ', Relaxed'
    plt.title(title)
    plt.savefig(os.path.join(os.environ["WORKDIR"], run_dir, "scan_result.png"))
    plt.show()



if __name__ == "__main__":
    # put here the function that you actually want to run
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('potential', type=str, help='Potential to use: eam or lj')
    parser.add_argument('--relax', action='store_true', help='Perform lattice relaxation instead of scanning')



    args = parser.parse_args()
    
    #check args to get appropriate input templates and run directories
    if args.potential == 'eam':
        cohesion = eam_cohesion
        true_alat = EAM_alat
        if args.relax:
            input_file = "eam_relax.in"
            run_dir = "Lab1/Vacancy/EAM/relax"
        else:
            input_file = "eam.in"
            run_dir = "Lab1/Vacancy/EAM/scan"
    elif args.potential == 'lj':
        cohesion = lj_cohesion
        true_alat = LJ_alat
        if args.relax:
            input_file = "lj_relax.in"
            run_dir = "Lab1/Vacancy/LJ/relax"
        else:
            input_file = "lj.in"
            run_dir = "Lab1/Vacancy/LJ/scan"
    with open(input_file, 'r') as f:
        input_template = f.read()

        lattice_sizescan()
