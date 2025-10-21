import numpy, os
import matplotlib.pyplot as plt
from labutil.plugins.pwscf import run_qe_pwscf, PWscf_inparam
from labutil.objects import Struc, Dir, ase2struc, Kpoints, Constraint, PseudoPotential, File
from ase.io import write
from ase import Atoms
from ase.build import bulk
def parse_qe_pwscf_output(outfile):
    cell_parameters = None #for scf calculations when cell parameters are never reported, so it doesnt freak out
    positions = None
    total_force = None
    walltime = None
    total_energy = None
    pressure = None
    volume = None
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



def make_struc( alloy, alat,):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :param alloy: can be 'Cu', 'Au' or 'CuAu'
    :return: structure object converted from ase
    """
    if alloy == "Cu" or alloy == "Au":
        shape = "fcc"
        cell = bulk(alloy, shape, a=alat)
        print(cell, cell.get_atomic_numbers())
        structure = Struc(ase2struc(cell))
        return structure
    elif alloy == "CuAu":
        # CuAu in L1_0 structure
        a = alat
        c = alat * 1.385  # approximate c/a ratio for CuAu L1_0
        cell = Atoms(
            symbols=["Cu", "Au"],
            positions=[[0, 0, 0], [a / 2, a / 2, c / 2]],
            cell=[ [a, 0, 0], [0, a, 0], [0, 0, c] ],
            pbc=True,
        )
        print(cell, cell.get_atomic_numbers())
        structure = Struc(ase2struc(cell))
        return structure


def compute_energy(alat, ecut, nk, alloy, recalculate=False):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    pseudopots = {
        "Cu": PseudoPotential(
            ptype="uspp", element="Cu", functional="LDA", name="Cu.pz-d-rrkjus.UPF"
        ),
        "Au": PseudoPotential(
            ptype="uspp", element="Au", functional="LDA", name="Au.pz-d-rrkjus.UPF"
        ),
    }
    struc = make_struc(alat=alat, alloy=alloy)
    # fix the Cu and Au atoms in place during relaxation
    constraint = Constraint(atoms={"0": [0, 0, 0], "1": [0, 0, 0]})
    kpts = Kpoints(gridsize=[nk, nk, nk], option="automatic", offset=True)
    dirname = "{}_nk_{}".format(alloy, nk)
    runpath = Dir(path=os.path.join(os.environ["WORKDIR"], "Lab4/Problem3", dirname))
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
                "calculation": "vc-relax",
                "pseudo_dir": os.environ["QE_POTENTIALS"],
                "outdir": runpath.path,
                "tstress": True,
                "tprnfor": True,
                "disk_io": "none",
            },
            "SYSTEM": {
                "ecutwfc": ecut,
                "ecutrho": ecut * 8,
                "occupations": "smearing",
                "smearing": "mp",
                "degauss": 0.02,
            },
            "ELECTRONS": {
                "diagonalization": "david",
                "mixing_beta": 0.7,
                "conv_thr": 1e-7,
            },
            "IONS": {"ion_dynamics": "bfgs"},
            "CELL": {"cell_dynamics": "bfgs"},
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

def lattice_scan(alloy):
    nks = [4, 6, 8, 10, 12, 14, 16]
    ecut = 40
    alat = 3.6 if alloy == "Cu" else 4.0 if alloy == "Au" else 3.8
    energy_list = []
    alats = []
    for nk in nks:
        output = compute_energy(alat=alat, ecut=ecut, nk=nk, alloy=alloy)
        energy_list.append(output["energy"])
        alat = output["cell_parameters"][0][1] * 2  # update alat based on relaxed cell parameter
        alats.append(alat)
          # assuming cubic cell, take a from first vector
        print(output)
    E_true = energy_list[-1]  # assuming the last one is the most converged
    data = {
        "K-mesh parameter": nks,
        "Energy (eV) per atom": energy_list,
        "Convergence Level (eV) (per atom)": [abs(e - E_true) for e in energy_list],
        "Relaxed Lattice parameter (Angstrom)": alats
    }
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv("nk_convergence_{}.csv".format(alloy), index=False)


    plt.plot(nks, energy_list, label=alloy)
    plt.xlabel("K-point Grid Size")
    plt.ylabel("Energy (eV) per atom")
    plt.title("K-point Grid Size Scan for {}".format(alloy))
    plt.grid()
    plt.legend()
    plt.savefig("nk_scan_{}.png".format(alloy))
    plt.show()


if __name__ == "__main__":
    # put here the function that you actually want to run
    lattice_scan("Cu")
    lattice_scan("Au")