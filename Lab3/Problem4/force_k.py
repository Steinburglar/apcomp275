import os
from labutil.plugins.pwscf import run_qe_pwscf, PWscf_inparam, parse_qe_pwscf_output
from labutil.objects import Struc, Dir, ase2struc, Kpoints, PseudoPotential
from ase.spacegroup import crystal
from ase.io import write
import matplotlib.pyplot as plt
import time
import re


def parse_k_points_from_output(output_file_path):
    """
    Parse an output file to find the line with 'number of k points' and return the value.
    
    Args:
        output_file_path (str): Path to the output file
    
    Returns:
        int: The number of k points, or None if not found
    """
    try:
        with open(output_file_path, 'r') as file:
            for line in file:
                # Look for line containing "number of k points"
                if 'number of k points' in line.lower():
                    # Extract the number using regex
                    match = re.search(r'number of k points\s*[=:]\s*(\d+)', line, re.IGNORECASE)
                    if match:
                        return int(match.group(1))
                    # Alternative pattern in case the format is different
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        return int(numbers[-1])  # Return the last number found
        return None
    except FileNotFoundError:
        print(f"Error: File {output_file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None




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
    #perturb the position of the atom slightly
    gecell.positions[0, 2] = gecell.positions[0, 2] + 0.05 * alat
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
    runpath = Dir(path=os.path.join(script_dir, f"k_{nk}"))
    input_params = PWscf_inparam(
        {
            "CONTROL": {
                "calculation": "scf",
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
    kpts = parse_k_points_from_output(output_file.path)
    output['k_points'] = kpts
    return output


def lattice_scan(nk):
    #params : ecut - cutoff radius
    nk = nk
    ecut = 30
    alat = 5.658

    #record time for each calculation
    a = time.time()
    output = compute_energy(alat=alat, ecut=ecut, nk=nk)
    b = time.time()
    time_ = b - a
    force = output["force"]
    kpts = output["k_points"]
    print(force)
    return force, time_, kpts



if __name__ == "__main__":
    # put here the function that you actually want to run
    nks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    forces = []
    times = []
    kpoints = []
    for nk in nks:
        f, t, k = lattice_scan(nk)
        forces.append(f)
        times.append(t)
        kpoints.append(k)
    #create a table with cuts, energy in eV, energy in Ry, and difference with the last value, which we call E_true
    F_true = forces[-1]
    import pandas as pd

    data = {
        "K-points": nks,
        "Total Force (eV/Angstrom)": forces,
        "Convergence Level (eV/Angstrom), Total": [abs(f - F_true) for f in forces],
        "Time (s)": times,
        "Unique k": kpoints,
    }
    df = pd.DataFrame(data)
    df.to_csv('cutoff_convergence_results.csv', index=False)
    print(df)
    plt.plot(nks, forces, marker='o')
    plt.title('K-point Sampling Force Convergence')
    plt.xlabel('K-points')
    plt.ylabel('Total Force (eV/Angstrom)')
    plt.savefig('force_k.png')
    plt.show()
