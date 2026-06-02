from __future__ import annotations

from typing import Any

def parse_molden_to_dict(filepath: str) -> dict[str, Any]:
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