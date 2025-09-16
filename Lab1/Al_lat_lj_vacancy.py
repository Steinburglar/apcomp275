from labutil.plugins.lammps import lammps_run, get_lammps_energy
from labutil.objects import Struc, Dir, ClassicalPotential, ase2struc
from ase.spacegroup import crystal
from ase.build import make_supercell
import numpy, os
import matplotlib.pyplot as plt
import argparse

true_alat = 4.05

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


def make_struc(alat, size = 1):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    unitcell = crystal(
        "Al", [(0, 0, 0)], spacegroup=225, cellpar=[alat, alat, alat, 90, 90, 90]
    )
    multiplier = numpy.identity(3) * size
    ase_supercell = make_supercell(unitcell, multiplier)
    structure = Struc(ase2struc(ase_supercell))
    return structure


def compute_vacancy_energy(alat, template, size = 1, super = False, relax = False, ):
    Eperf = compute_energy(alat, template, size=size, super=super, relax=relax, structure = "perfect")[0]
    Evac = compute_energy(alat, template, size=size, super=super, relax=relax, structure = "vacancy")[0]
    n = size**3 * 4  #number of atoms in supercell


def compute_energy(alat, template, size = 1, relax = False, structure= "perfect"):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    run_dir = "Lab1/vacancy/" +structure
    if relax:
        run_dir += "/relax"
    else:
        run_dir += "/still"

    potpath = os.path.join(os.environ["LAMMPS_POTENTIALS"], "Al_zhou.eam.alloy")  #path to desired potential function
    potential = ClassicalPotential(path=potpath, ptype="eam", element=["Al"])   #define potential object
    runpath = Dir(path=os.path.join(os.environ["WORKDIR"], run_dir, str(size)))  #pass a directory object to variable runpath
    struc = make_struc(alat=alat, size = size)   #builds structure object based on lattice parameter
    output_file = lammps_run(
        struc=struc,
        runpath=runpath,
        potential=None, #not sure syntax here, but making sure eam does not get used
        intemplate=template,
        inparam={},
    )
    energy, lattice = get_lammps_energy(outfile=output_file)   #labutils function. lattice is just lattice constant
    return energy, lattice


def lattice_scan_supersize(relax=False):
    size_list = numpy.arange(1, 5)
    energy_list = [
        compute_vacancy_energy(alat=true_alat, template=input_template, size=s, super=True, relax=relax)[0] for s in size_list
    ]
    plt.plot(size_list, energy_list)
    title = 'Supercell size Vacancy Scan Result, LJ'
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
