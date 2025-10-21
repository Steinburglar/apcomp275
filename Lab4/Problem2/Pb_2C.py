import numpy, os
import pandas as pd
import matplotlib.pyplot as plt
from labutil.plugins.pwscf import run_qe_pwscf, PWscf_inparam, parse_qe_pwscf_output
from labutil.objects import Struc, Dir, ase2struc, Kpoints, Constraint, PseudoPotential, File
from ase.io import write
from ase import Atoms


def parse_qe_pwscf_output(outfile):
    cell_parameters = None #for scf calculations when cell parameters are never reported, so it doesnt freak out
    positions = None
    with open(outfile.path, "r") as outf:
        for line in outf:
            if line.lower().startswith("     pwscf"):
                walltime = line.split()[-3] + line.split()[-2]
            if line.lower().startswith("     total force"):
                total_force = float(line.split()[3]) * (13.605698066 / 0.529177249)
            if line.lower().startswith("!    total energy"):
                total_energy = float(line.split()[-2]) * 13.605698066
            if line.lower().startswith("          total   stress"):
                pressure = float(line.split()[-1])
            if line.lower().startswith("cell_parameters"):
                # we could extract cell parameters here if needed
                cell_parameters = []
                for _ in range(3):
                    line = next(outf)
                    cell_parameters.append([float(x) for x in line.split()])
            if line.lower().startswith("     unit-cell volume"):
                volume = float(line.split()[-2])  # in Au^3
                volume = volume * 0.529177249**3  # convert to Angstrom^3
            if line.startswith("ATOMIC_POSITIONS"):
                # grab the next 5 lines verbatim, preserving all whitespace and newlines
                positions = []
                for _ in range(5):
                    positions.append(next(outf))
    result = {
        "energy": total_energy,
        "force": total_force,
        "pressure": pressure,
        "walltime": walltime,
        "cell_parameters": cell_parameters,
        "volume": volume,
        "positions": positions
    }
    return result


def make_struc(alat, displacement=0):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    lattice = alat * numpy.identity(3)
    symbols = ["Pb", "Ti", "O", "O", "O"]
    sc_pos = [
        [0, 0, 0],
        [0.5, 0.5, 0.5 + displacement],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    perov = Atoms(symbols=symbols, scaled_positions=sc_pos, cell=lattice)
    # check how your cell looks like
    # write('s.cif', perov)
    structure = Struc(ase2struc(perov))
    return structure


def compute_energy(alat, nk, ecut, displ=0, recalculate=False):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    pseudopots = {
        "Pb": PseudoPotential(
            ptype="uspp", element="Pb", functional="LDA", name="Pb.pz-d-van.UPF"
        ),
        "Ti": PseudoPotential(
            ptype="uspp", element="Ti", functional="LDA", name="Ti.pz-sp-van_ak.UPF"
        ),
        "O": PseudoPotential(
            ptype="uspp", element="O", functional="LDA", name="O.pz-rrkjus.UPF"
        ),
    }
    struc = make_struc(alat=alat, displacement=displ)
    # fix the Pb and Ti atoms in place during relaxation
    constraint = Constraint(atoms={"0": [0, 0, 0]}) # only constrain Pb
    kpts = Kpoints(gridsize=[nk, nk, nk], option="automatic", offset=True)
    dirname = "PbTiO3_a_{}_ecut_{}_nk_{}_displ_{}_RelaxTi".format(alat, ecut, nk, displ)
    runpath = Dir(path=os.path.join(os.environ["WORKDIR"], "Lab4/Problem2", dirname))
    # Check if calculation already exists and is complete
    if not recalculate and os.path.exists(runpath.path):
        existing_output = check_existing_calculation(runpath)
        if existing_output is not None:
            print(f"Found existing completed calculation in {runpath.path}")
            output = parse_qe_pwscf_output(outfile=existing_output)
            return output
        else:
            print(f"Existing calculation found but incomplete in {runpath.path}, rerunning...")

    input_params = PWscf_inparam(
        {
            "CONTROL": {
                "calculation": "relax",
                "pseudo_dir": os.environ["QE_POTENTIALS"],
                "outdir": runpath.path,
                "tstress": True,
                "tprnfor": True,
                "disk_io": "none",
            },
            "SYSTEM": {
                "ecutwfc": ecut,
                "ecutrho": ecut * 8,
            },
            "ELECTRONS": {
                "diagonalization": "david",
                "mixing_beta": 0.7,
                "conv_thr": 1e-7,
            },
            "IONS": {"ion_dynamics": "bfgs"},
            "CELL": {},
        }
    )

    output_file = run_qe_pwscf(
        runpath=runpath,
        struc=struc,
        pseudopots=pseudopots,
        params=input_params,
        kpoints=kpts,
        constraint=constraint,
        ncpu=2,
    )
    output = parse_qe_pwscf_output(outfile=output_file)
    return output

def check_existing_calculation(runpath):
    """
    Check if pwscf.out exists in the given directory.
    
    Parameters:
    runpath: Dir object with path to check
    
    Returns:
    File object pointing to pwscf.out if it exists, None otherwise
    """
    output_file_path = os.path.join(runpath.path, "pwscf.out")
    
    if os.path.exists(output_file_path):
        return File({"path": output_file_path})
    
    return None

def lattice_scan():
    nk = 4
    ecut = 30
    alat = 3.875
    init_displ = 0.025
    output = compute_energy(alat=alat, ecut=ecut, nk=nk, displ=init_displ)
    energy = output["energy"]
    positions = output["positions"]
    print(positions)
    #save output energy and cell_parameters to a file
    with open("relaxed_structure.txt", "w") as f:
        f.write(f"Total Energy (eV): {energy}\n")
        f.write("Atomic Positions (Angstrom):\n")
        for p in positions:
            print(p)
            f.write(p + "\n")



if __name__ == "__main__":
    # put here the function that you actually want to run
    lattice_scan()
    #runs have found 3.875 as a good alat