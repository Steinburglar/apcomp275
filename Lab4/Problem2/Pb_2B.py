import numpy, os
import matplotlib.pyplot as plt
from labutil.plugins.pwscf import run_qe_pwscf, PWscf_inparam, parse_qe_pwscf_output
from labutil.objects import Struc, Dir, ase2struc, Kpoints, Constraint, PseudoPotential, File
from ase.io import write
from ase import Atoms


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
    constraint = Constraint(atoms={"0": [0, 0, 0], "1": [0, 0, 0]})
    kpts = Kpoints(gridsize=[nk, nk, nk], option="automatic", offset=True)
    dirname = "PbTiO3_a_{}_ecut_{}_nk_{}_displ_{}".format(alat, ecut, nk, displ)
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
    #displ_list = numpy.arange(-0.05, 0.05, 0.01)
    displ_list = [-0.05, -0.04, -0.035, -0.03, -0.025, -0.02, -0.01, 0.0, 0.01, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.035, 0.04]
    print(displ_list)
    energy_list = []
    for displ in displ_list:
        output = compute_energy(alat=alat, ecut=ecut, nk=nk, displ=displ)
        energy_list.append(output["energy"])
        print(output)
    print(displ_list)
    print(energy_list)
    #find minimum energy and corresponding displacement
    min_energy = min(energy_list)
    min_index = energy_list.index(min_energy)
    optimal_displ = displ_list[min_index]
    print(f"Minimum energy: {min_energy} eV at displacement: {optimal_displ} Å")
    #plot energy vs displacement
    plt.plot(displ_list, energy_list)
    plt.xlabel("Displacement (Å)")
    plt.ylabel("Energy (eV) per atom")
    plt.title("Displacement Scan")
    plt.grid()
    plt.savefig("displ_scan.png")
    plt.show()


if __name__ == "__main__":
    # put here the function that you actually want to run
    lattice_scan()
    #runs have found 3.875 as a good alat