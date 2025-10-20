"""Example of running energy calculations on a simple Ge system with Quantum Espresso *using the ASE calculator interface*."""

import os

import ase
import ase.spacegroup
import ase.io
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.units import create_units

# ASE's `ase.units` is a useful tool for working with unit conversions and physical constants
# It allows us to get units from various different iterations of standards.
# In this script, we follow ASE in using the CODATA 2006 units, since
# "Quantum ESPRESSO uses CODATA 2006 internally" according to ASE:
# https://gitlab.com/ase/ase/-/blob/master/ase/io/espresso.py#L37
units = create_units("2006")
# If you just want the default units (reasonable for most purposes) you can use
# ase.units.whatever yourself without `create_units`.


def make_struc(alat: float) -> ase.Atoms:
    """
    Creates the crystal structure using ASE.

    Args:
        alat: Lattice parameter in angstrom

    Returns:
        The crystal structure as an `ase.Atoms` object.
    """
    # set primitive_cell=False if you want to create a simple cubic unit cell with 8 atoms
    return ase.spacegroup.crystal(
        "Ge",
        [(0, 0, 0)],
        spacegroup=227,
        cellpar=[alat, alat, alat, 90, 90, 90],
        primitive_cell=True,
    )


def make_calculator(nk: int, ecut: float) -> Espresso:
    """Make an ASE Quantum Espresso "calculator" to call Quantum Espresso from ASE.

    See https://wiki.fysik.dtu.dk/ase/ase/calculators/espresso.html#module-ase.calculators.espresso
    for documentation and more details on the calculator object and its behavior.

    Args:
        nk (int): dimension of the k-point grid
        ecut (float): the energy cutoff
    """
    # We need to tell ASE where to find `pw.x` (the Quantum Espresso command)
    # since it is in a specific place we put it:
    profile = EspressoProfile(
        command=f"mpirun -np 2 {os.environ['QE_PW_COMMAND']}",
        pseudo_dir=os.environ["QE_POTENTIALS"],
    )
    #       ^       ^               ^                |___________________________|
    #       |       |               |                 \- from ASE's espresso.py default command
    #       |       |               |                    PREFIX is replaced by ASE with the name of ASE's files
    #       |       |               |
    #       |       |               \-  Where our pw.x is
    #       |       |
    #       |       \-  Run on two cores / ranks (our default VM has this many cores)
    #       |
    #       \-  Run QE on multiple CPU cores using MPI

    workdir = os.path.join(os.environ["WORKDIR"], "Lab3", "test")
    os.makedirs(workdir, exist_ok=True)

    return Espresso(
        profile=profile,
        # Specify, using this special options for ASE, which pseudopotentials to use for each kind of atom:
        pseudopotentials={"Ge": "ge_lda_v1.4.uspp.F.UPF"},
        # Tell ASE where to tell QE to find pseudopotentials:
        # (kept for clarity; profile.pseudo_dir is already set)
        pseudo_dir=os.environ["QE_POTENTIALS"],
        # ASE has a special option for setting up the k-point mesh:
        # (See https://wiki.fysik.dtu.dk/ase/ase/calculators/espresso.html#parameters)
        kpts=(nk, nk, nk),
        # Note also that options related to input and output, such as
        # `outdir`, which we set in the labutil version of this example,
        # are not set: ASE handles input and output management automatically
        # as part of the calculator. Instead, we just tell ASE where we want it to run:
        directory=workdir,
        # You can change this ^ on the calculator later (`calc.directory = "/something/somewhere/"`) if you want to.
        # !!! If something goes wrong, you can look at `espresso.pwo` in `directory` to see the output of QE !!!
        #
        # Finally, all other options to QE can be transparently specified in `input_data`,
        # just as they would be in the QE input file. From the ASE docs:
        #
        #     All parameters must be given in QE units, usually Ry or atomic units in line
        #     with the documentation. ASE does not add any defaults over the defaults of QE.
        #     Parameters can be given as keywords and the calculator will put them into the
        #     correct section of the input file.
        #
        # Note that capital letters in the section names are not needed.
        input_data={
            "control": {
                "tstress": True,
                "tprnfor": True,
            },
            "system": {
                "ecutwfc": ecut,
            },
            "electrons": {
                "diagonalization": "david",
                "mixing_beta": 0.5,
                "conv_thr": 1e-7,
            },
        },
    )


def lattice_scan():
    nk = 3
    ecut = 30
    alat = 5.0
    # We create the calculator once:
    calc = make_calculator(nk=nk, ecut=ecut)
    for alat in [5.0]:
        atoms = make_struc(alat=alat)
        # And assign it to be used by each `ase.Atoms` we are interested in:
        atoms.calc = calc
        # The first call to a calculator-related property will trigger ASE to call the calculator:
        energy = atoms.get_potential_energy()
        # Further calls like `atoms.get_forces()` here will use the cached results of the first call.

        # Note that ASE converts results from the calculator back into its own standard system of units,
        # which is eV (electronvolts) for energy and Angstroms for positions.
        # (labutil does the same, see `parse_qe_pwscf_output` in labutil/plugins/pwscf.py)
        print(f"Energy from QE (eV): {energy}")
        # We can also convert back into QE's usual standard units (Rydbergs).
        # ASE's `units` is set up so that `3 * units.Ry`, for example, gives the equivalent
        # in ASE's own eV. In other words, `units.Ry` has units `eV / Ry`, so to convert backward
        # we need to divide (multiply [eV] * [Ry / eV]):
        print(f"Energy from QE (Ry): {energy / units.Ry}")


if __name__ == "__main__":
    # put here the function that you actually want to run
    lattice_scan()