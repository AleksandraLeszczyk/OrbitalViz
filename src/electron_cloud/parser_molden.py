import re
from typing import Dict, List, Any

def parse_molden_to_dict(filepath: str) -> Dict[str, Any]:
    """
    Parses a standard Molden file to extract atomic coordinates and basis set (GTO) details.
    
    Returns a dictionary perfectly formatted for manual construction of PyBEST 
    or custom quantum chemistry basis set objects.
    """
    
    # Initialize the requested data structure
    data = {
        "atoms": [],                 # List[int]: Atomic numbers
        "coordinates": [],           # List[List[float]]: XYZ coordinates
        "number_of_primitives": [],  # List[int]: Primitives per shell
        "contraction": [],           # List[float]: Contraction coefficients
        "alpha": [],                 # List[float]: Exponents
        "shell_types": [],           # List[int]: Angular momentum (s=0, p=1, d=2...)
        "shell_to_atom": []          # List[int]: 0-based atom index for each shell
    }

    # Angular momentum string-to-int mapping
    shell_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}

    current_section = None
    current_atom_idx = -1

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        # Skip empty lines or standard comments
        if not line or line.startswith('#'):
            continue

        # Detect section headers (e.g., [Atoms], [GTO], [MO])
        if line.startswith('['):
            header = line.lower()
            if '[atoms]' in header:
                current_section = 'atoms'
            elif '[gto]' in header:
                current_section = 'gto'
            else:
                current_section = 'other' # We ignore [MO], [5D], [5D7F], etc.
            continue

        # Process Atoms Section
        if current_section == 'atoms':
            # Molden atoms line: Element, Sequence_Num, Atomic_Num, x, y, z
            parts = line.split()
            if len(parts) >= 6:
                data["atoms"].append(int(parts[2]))
                data["coordinates"].append([
                    float(parts[3]), 
                    float(parts[4]), 
                    float(parts[5])
                ])

        # Process Basis Set (GTO) Section
        elif current_section == 'gto':
            parts = line.split()

            # Detect Atom Header in GTO (format: "Atom_Index 0")
            if len(parts) == 2 and parts[1] == '0' and parts[0].isdigit():
                # Convert from 1-based Molden indexing to 0-based Python indexing
                current_atom_idx = int(parts[0]) - 1
                continue

            # Detect Shell Header (format: "Shell_Type Num_Primitives Scale_Factor")
            if len(parts) == 3 and parts[0].lower() in shell_map:
                shell_type_str = parts[0].lower()
                num_primitives = int(parts[1])

                data["shell_types"].append(shell_map[shell_type_str])
                data["number_of_primitives"].append(num_primitives)
                data["shell_to_atom"].append(current_atom_idx)

                # Iterate through the primitives for this shell
                for _ in range(num_primitives):
                    if i >= len(lines):
                        break
                    
                    prim_line = lines[i].strip()
                    i += 1
                    
                    if not prim_line:
                        continue 
                        
                    # Handle Fortran double precision "D" format (e.g., 1.0D+01 -> 1.0E+01)
                    prim_parts = prim_line.replace('D', 'E').replace('d', 'e').split()

                    data["alpha"].append(float(prim_parts[0]))
                    data["contraction"].append(float(prim_parts[1]))

    return data

import numpy as np


def read_molden_c_matrix(filepath: str) -> np.ndarray:
    """
    Reads the AO/MO transformation matrix directly from a Molden file using pure Python and NumPy.
    
    Args:
        filepath (str): Path to the .molden file.
        
    Returns:
        np.ndarray: The transformation matrix C of shape (N_AO, N_MO).
    """
    mo_matrix = []
    current_mo = []
    in_mo_section = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check if we are entering the MO section
            if line.upper().startswith('[MO]'):
                in_mo_section = True
                continue

            # If we hit another bracketed section, we are done with MOs
            if line.startswith('['):
                if in_mo_section:
                    if current_mo:
                        mo_matrix.append(current_mo)
                    break

            if in_mo_section:
                # Metadata lines contain '=' or alphabetic characters (Sym=, Ene=, Alpha, etc.)
                if '=' in line or line.isalpha():
                    # If we already collected AOs for an MO, save it before starting the new one
                    if current_mo:
                        mo_matrix.append(current_mo)
                        current_mo = []
                    continue

                # Parse the AO index and coefficient
                parts = line.split()
                if len(parts) == 2:
                    try:
                        # Molden often uses Fortran-style scientific notation (e.g., 0.123D-02)
                        # We must convert 'D' or 'd' to 'E' for Python's float() to understand it.
                        coeff_str = parts[1].replace('D', 'E').replace('d', 'e')
                        coeff = float(coeff_str)
                        current_mo.append(coeff)
                    except ValueError:
                        continue

    # Catch the very last MO in the file
    if current_mo:
        mo_matrix.append(current_mo)

    # mo_matrix is currently a list of MOs, meaning shape is (N_MO, N_AO).
    # The standard quantum chemical convention for the C matrix is (N_AO, N_MO).
    # We convert to a NumPy array and transpose it.
    c_matrix = np.array(mo_matrix).T
    
    return c_matrix