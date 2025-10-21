import numpy, os
import matplotlib.pyplot as plt
from labutil.plugins.pwscf import run_qe_pwscf, PWscf_inparam
from labutil.objects import Struc, Dir, ase2struc, Kpoints, PseudoPotential, File
from ase.spacegroup import crystal
from ase.io import write
from ase.build import bulk


def parse_qe_pwscf_output(outfile):
    cell_parameters = None #for scf calculations when cell parameters are never reported, so it doesnt freak out
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
    result = {
        "energy": total_energy,
        "force": total_force,
        "pressure": pressure,
        "walltime": walltime,
        "cell_parameters": cell_parameters,
        "volume": volume,
    }
    return result



def make_struc(alat, form, clat = None, anti = False):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :param form: Crystal form, e.g. 'bcc', 'hcp'
    :param clat: c lattice parameter for hcp
    :param anti: Boolean flag to include a fake Co in the structure, to allow for anti-ferromagnetic ordering
    :return: structure object converted from ase
    """
    fecell = bulk("Fe", form, a=alat, c=clat)
    # check how your cell looks like
    # write('s.cif', gecell)
    print(fecell, fecell.get_atomic_numbers())
    if anti:
        fecell.set_atomic_numbers([26, 27])
    structure = Struc(ase2struc(fecell))
    print(structure.species)
    return structure


def compute_energy(form, alat, nk, ecut, clat=None, anti=False, recalculate=False):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    potname = "Fe.pbe-nd-rrkjus.UPF"
    potpath = os.path.join(os.environ["QE_POTENTIALS"], potname)
    pseudopots = {
        "Fe": PseudoPotential(
            path=potpath, ptype="uspp", element="Fe", functional="GGA", name=potname
        ),
        "Co": PseudoPotential(
            path=potpath, ptype="uspp", element="Fe", functional="GGA", name=potname
        ),
    }
    struc = make_struc(alat=alat, form=form, clat=clat, anti=anti)
    if form == "bcc":
        kpts = Kpoints(gridsize=[nk, nk, nk], option="automatic", offset=False)
    elif form == "hcp":
        kpts = Kpoints(gridsize=[nk, nk, int(nk/2)], option="automatic", offset=False)
    else:
        raise ValueError("Form not recognized")
    dirname = "Fe_{}_alat_{}".format(form, alat)
    runpath = Dir(path=os.path.join(os.environ["WORKDIR"], "Lab4/1B", dirname))

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
                "calculation": "scf",
                "pseudo_dir": os.environ["QE_POTENTIALS"],
                "outdir": runpath.path,
                "tstress": True,
                "tprnfor": True,
                "disk_io": "none",
            },
            "SYSTEM": {
                "ecutwfc": ecut,
                "ecutrho": ecut * 10,
                "nspin": 2,
                "starting_magnetization(1)": 0.7,
                "occupations": "smearing",
                "smearing": "mp",
                "degauss": 0.02,
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
    if anti:
        input_params["SYSTEM"]["starting_magnetization(1)"] = 1
        input_params["SYSTEM"]["starting_magnetization(2)"] = -1

    output_file = run_qe_pwscf(
        runpath=runpath,
        struc=struc,
        pseudopots=pseudopots,
        params=input_params,
        kpoints=kpts,
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


def extract_lattice_parameters(cell_vectors, crystal_shape):
    """
    Extract lattice parameters from QE cell vectors (ASE primitive cells).
    
    Parameters:
    cell_vectors: list of 3 lists, each with 3 floats (the lattice vectors)
    crystal_shape: str, crystal structure type ('bcc', 'hcp')
    
    Returns:
    dict with relevant lattice parameters
    """
    import numpy as np
    
    a1, a2, a3 = [np.array(v) for v in cell_vectors]
    
    if crystal_shape.lower() == 'bcc':
        # BCC primitive: a1 = (a/2)*(-1,1,1), etc.
        # Lattice parameter: a = 2 * |a1[0]|
        alat = 2 * abs(a1[0])
        return {'a': alat}
        
    elif crystal_shape.lower() == 'hcp':
        # HCP: a1 = (a,0,0), a2 = (-a/2, a*sqrt(3)/2, 0), a3 = (0,0,c)
        # a-parameter: a = a1[0]
        # c-parameter: c = a3[2]  
        alat = a1[0]
        clat = a3[2]
        return {'a': alat, 'c': clat}
        

    else:
        raise ValueError(f"Crystal shape '{crystal_shape}' not implemented")


def lattice_scan(form, alat, anti=False):
    #right now flexible onform of iron and number of kpoints
    nk = 14
    ecut = 30
    alat = alat #halfway between expected bcc and hcp
    clat = alat * 1.737 #calculated clat/alat ratio for hcp
    output = compute_energy(form, alat, nk, ecut, clat=clat, anti=anti)
    print(output)
    energy = output["energy"]
    volume = output["volume"] ##need to parse this from output
    return energy, volume

if __name__ == "__main__":
    # put here the function that you actually want to run
    alats_bcc = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6][:6]
    alats_hcp = [ 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4][:6]
    bcc_energies = []
    bcc_volumes = []
    hcp_energies = []
    hcp_volumes = []
    
    for alat in alats_bcc:
        e, v = lattice_scan("bcc", alat)
        bcc_energies.append(e)
        bcc_volumes.append(v)
    for alat in alats_hcp:
        e, v = lattice_scan("hcp", alat)
        hcp_energies.append(e)
        hcp_volumes.append(v) 
    #normalize hcp by number of atoms: 2
    for i in range(len(hcp_energies)):
        hcp_energies[i] /= 2
        hcp_volumes[i] /= 2
    #create a table with nks (the mesh spacing), energy in eV, energy in Ry, and difference with the last value, which we call E_true
    energies_ry_bcc = [e * 0.0684286 for e in bcc_energies]
    energies_ry_hcp = [e * 0.0684286 for e in hcp_energies]
    E_true_bcc = bcc_energies[-1]
    E_true_hcp = hcp_energies[-1]
    import pandas as pd

    bcc_data = {
        "Volume per atom": bcc_volumes,
        "Energy (eV) per atom": bcc_energies,
        "Energy (Ry) per atom": energies_ry_bcc,
        "relaxed a (Angstrom)": alats_bcc,
        "Convergence Level (eV) (per atom)": [abs(e - E_true_bcc)for  e in bcc_energies],
        "Convergence Level (Ry)(per atom) ": [abs(e - E_true_bcc * 0.0684286) for e in energies_ry_bcc],
    }
    hcp_data = {
        "Volume per atom": hcp_volumes,
        "Energy (eV) per atom": hcp_energies,
        "Energy (Ry) per atom": energies_ry_hcp,
        "a (Angstrom)": alats_hcp,
        "c (Angstrom)": [a *1.737 for a in alats_hcp],
        "Convergence Level (eV) (per atom)": [abs(e - E_true_hcp) for e in hcp_energies], #divide by 2 since 2 atoms per cell
        "Convergence Level (Ry)(per atom) ": [abs(e - E_true_hcp * 0.0684286) for e in energies_ry_hcp],
    }
    bcc_df = pd.DataFrame(bcc_data)
    hcp_df = pd.DataFrame(hcp_data)
    bcc_df.to_csv('k_convergence_bcc_results.csv', index=False)
    hcp_df.to_csv('k_convergence_hcp_results.csv', index=False)
    plt.plot(bcc_volumes, bcc_energies, marker='o', label='BCC')
    plt.title('BCC K parameter Convergence')
    plt.xlabel('Volume per atom (Angstrom^3)')
    plt.ylabel('Total Energy (eV) per atom')
    plt.savefig('bcc_convergence.png')
    plt.show()

    plt.plot(hcp_volumes, hcp_energies, marker='o', color='orange', label='HCP')
    plt.title('HCP K parameter Convergence')
    plt.xlabel('Volume per atom (Angstrom^3)')
    plt.ylabel('Total Energy (eV) per atom')
    plt.savefig('hcp_convergence.png')
    plt.show()


    # plot both overlapping:
    plt.plot(bcc_volumes, bcc_energies, marker='o', label='BCC')
    plt.plot(hcp_volumes, hcp_energies, marker='o', color='orange', label='HCP')
    plt.title('K parameter Convergence')
    plt.xlabel('Volume per atom (Angstrom^3)')
    plt.ylabel('Total Energy (eV) per atom')
    plt.legend()
    plt.grid()
    plt.savefig('bcc_hcp_convergence.png')
    plt.show()


    