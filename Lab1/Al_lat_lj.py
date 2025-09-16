from labutil.plugins.lammps import lammps_run, get_lammps_energy
from labutil.objects import Struc, Dir, ClassicalPotential, ase2struc
from ase.spacegroup import crystal
from ase.build import make_supercell
import numpy, os
import matplotlib.pyplot as plt
import argparse


input_template = """
# ---------- 1. Initialize simulation ---------------------
units metal
atom_style atomic
dimension  3
boundary   p p p
read_data $DATAINPUT

# ---------- 2. Specify interatomic potential ---------------------

pair_style lj/cut 4.5
pair_coeff 1 1 0.392 2.620 4.5

# ---------- 3. Run single point calculation  ---------------------
thermo_style custom step pe lx ly lz press pxx pyy pzz
run 0

# ---- 4. Define and print useful variables -------------
variable natoms equal "count(all)"
variable totenergy equal "pe"
variable length equal "lx"

print "Total energy (eV) = ${totenergy}"
print "Number of atoms = ${natoms}"
print "Lattice constant (Angstoms) = ${length}"
"""


def make_struc(alat, super = False):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    unitcell = crystal(
        "Al", [(0, 0, 0)], spacegroup=225, cellpar=[alat, alat, alat, 90, 90, 90]
    )
    if super:
        multiplier = numpy.identity(3) * 2
        ase_supercell = make_supercell(unitcell, multiplier)
        structure = Struc(ase2struc(ase_supercell))
    else:
        structure = Struc(ase2struc(unitcell))
    return structure


def compute_energy(alat, template, super = False, relax = False):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    if super:
        run_dir = "Lab1/LJsuper"
    else:
        run_dir = "Lab1/LJ"
    if relax:
        run_dir += "/relax"
    else:
        run_dir += "/scan"

    potpath = os.path.join(os.environ["LAMMPS_POTENTIALS"], "Al_zhou.eam.alloy")  #path to desired potential function
    potential = ClassicalPotential(path=potpath, ptype="eam", element=["Al"])   #define potential object
    runpath = Dir(path=os.path.join(os.environ["WORKDIR"], run_dir, str(alat)))  #pass a directory object to variable runpath
    struc = make_struc(alat=alat)   #builds structure object based on lattice parameter
    output_file = lammps_run(
        struc=struc,
        runpath=runpath,
        potential=None, #not sure syntax here, but making sure eam does not get used
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
    plt.plot(alat_list, energy_list)
    title = 'Lattice Scan Result, LJ'
    if args.super:
        title += ', Supercell'
    plt.title(title)
    plt.show()

def lattice_relax():
    a = 4.1
    energy = compute_energy(alat=a, template=input_template, super = args.super, relax = True)[0]
    plt.plot(a, energy, 'o')
    title = 'Lattice Relaxation Result, LJ'
    if args.super:
        title += ', Supercell'
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # put here the function that you actually want to run
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--relax', action='store_true', help='Perform lattice relaxation instead of scanning')
    parser.add_argument('--super', action='store_true', help='Use supercell')
    args = parser.parse_args()

    if args.relax:
        lattice_relax()
    else:
        lattice_scan()
