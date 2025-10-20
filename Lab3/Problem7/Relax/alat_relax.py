import os
from labutil.plugins.pwscf import run_qe_pwscf, PWscf_inparam, parse_qe_pwscf_output
from labutil.objects import Struc, Dir, ase2struc, Kpoints, PseudoPotential
from ase.spacegroup import crystal
from ase.io import write
import matplotlib.pyplot as plt
import numpy as np


def make_struc(alat):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    # set primitive_cell=False if you want to create a simple cubic unit cell with 8 atoms
    gecell = crystal(
        "Ge",
        [(0, 0, 0)],
        spacegroup=227,
        cellpar=[alat, alat, alat, 90, 90, 90],
        primitive_cell=True,
    )
    # check how your cell looks like
    # write('s.cif', gecell)
    structure = Struc(ase2struc(gecell))
    return structure


def compute_energy(alat, nk, ecut):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    potname = "ge_lda_v1.4.uspp.F.UPF"
    pseudopath = os.environ["QE_POTENTIALS"]
    potpath = os.path.join(pseudopath, potname)
    pseudopots = {
        "Ge": PseudoPotential(
            name=potname, path=potpath, ptype="uspp", element="Ge", functional="LDA"
        )
    }
    struc = make_struc(alat=alat)
    kpts = Kpoints(gridsize=[nk, nk, nk], option="automatic", offset=False)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runpath = Dir(path=os.path.join(script_dir, "alat_relax"))
    input_params = PWscf_inparam(
        {
            "CONTROL": {
                "calculation": "vc-relax",
                "pseudo_dir": pseudopath,
                "outdir": runpath.path,
                "tstress": True,
                "tprnfor": True,
                "disk_io": "none",
            },
            "SYSTEM": {
                "ecutwfc": ecut,
            },
            "ELECTRONS": {
                "diagonalization": "david",
                "mixing_beta": 0.5,
                "conv_thr": 1e-7,
            },
            "IONS": {},
            "CELL": {},
        }
    )

    output_file = run_qe_pwscf(
        runpath=runpath,
        struc=struc,
        pseudopots=pseudopots,
        params=input_params,
        kpoints=kpts,
    )
    output = parse_qe_pwscf_output(outfile=output_file)
    return output


def lattice_relax():
    """
    Function to relax the lattice parameter of Ge.
    :return: energy, lattice param at relaxed lattice parameter"""
    #params : ecut - cutoff radius
    nk = 10 # from force convergence
    ecut = 40 # from force and e_Diff convergence
    alat0 = 5.65 # initial guess for lattice parameter in angstrom
    output = compute_energy(alat=alat0, ecut=ecut, nk=nk)
    energy = output["energy"]
    return energy, alat0

if __name__ == "__main__":
    #run the code to relax the lattice parameter
    lattice_relax()
    
