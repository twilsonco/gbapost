#!/usr/local/bin/python3

import csv, os
from datetime import datetime
import numpy as np
from sys import argv, exc_info

"""A permissive filename sanitizer."""
import unicodedata
import re



def sanitize(filename):
    """Return a fairly safe version of the filename.

    We don't limit ourselves to ascii, because we want to keep municipality
    names, etc, but we do want to get rid of anything potentially harmful,
    and make sure we do not exceed Windows filename length limits.
    Hence a less safe blacklist, rather than a whitelist.
    """
    blacklist = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", "\0"]
    reserved = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5",
        "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
        "LPT6", "LPT7", "LPT8", "LPT9",
    ]  # Reserved words on Windows
    filename = "".join(c for c in filename if c not in blacklist)
    # Remove all charcters below code point 32
    filename = "".join(c for c in filename if 31 < ord(c))
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.rstrip(". ")  # Windows does not allow these at end
    filename = filename.strip()
    if all([x == "." for x in filename]):
        filename = "__" + filename
    if filename in reserved:
        filename = "__" + filename
    if len(filename) == 0:
        filename = "__"
    if len(filename) > 255:
        parts = re.split(r"/|\\", filename)[-1].split(".")
        if len(parts) > 1:
            ext = "." + parts.pop()
            filename = filename[:-len(ext)]
        else:
            ext = ""
        if filename == "":
            filename = "__"
        if len(ext) > 254:
            ext = ext[254:]
        maxl = 255 - len(ext)
        filename = filename[:maxl]
        filename = filename + ext
        # Re-check last character (if there was no extension)
        filename = filename.rstrip(". ")
        if len(filename) == 0:
            filename = "__"
    return filename

class SGB_CONDENSED_BASIN_TYPE:
    MAX = 'max'
    MIN = 'min'
    MIXED = 'mixed'
    
def is_float(s):
    try:
        f=float(s)
        return np.isfinite(f)
    except:
        return False

def main():
    date_time_stamp = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    
    print(f"Starting GBA system comparison\n(using input: {' '.join(argv)})\n\n")
    
    ##################################
    # IMPORT FILES AND INITIAL PROCESSING
    ################################## 
    
    
    
    file_paths = [f for f in os.listdir(os.getcwd()) if ".csv" in f]
    
    print(f"Loading data from {len(file_paths)} file_paths in \"{os.getcwd()}\"...\n\n")

    # Import data
    all_input_files = [list(csv.DictReader(open(logfile, 'r'))) for logfile in file_paths]
    
    
    if len(all_input_files) == 0:
        print("No data found. Quitting...")
        return False
    
    # get list of systems
    systems = {}
    for s in all_input_files:
        s_name = s[0]['Dataset name']
        systems[s_name] = {}
    
    # get regions and their atomic components in each system
    for sys in systems.keys():
        atoms = {}
        regions = {}
        minmax_basins = {}
        for s in all_input_files:
            s_name = s[0]['Dataset name']
            if sys == s_name:
                for line in s:
                    a_name = line['Atom name'].split(':')[0].replace(' ','')
                    r_name = line['Region name']
                    if r_name.replace(' ','') != a_name:
                        if 'full' not in a_name:
                            if "Max" not in r_name and "Min" not in r_name:
                                if r_name not in regions:
                                    regions[r_name] = {}
                                regions[r_name][a_name] = {k:(float(v) if " (with boundary error)" not in k else v) for k,v in line.items() if is_float(v) and k not in ['Dataset name', 'Atom name', 'Region name']}
                            else:
                                r_name = r_name[:r_name.find(":")].strip().replace(' ','') + r_name[r_name.find(":"):]
                                minmax_basins[r_name] = {k:(float(v) if " (with boundary error)" not in k else v) for k,v in line.items() if is_float(v) and k not in ['Dataset name', 'Atom name', 'Region name']}
                    
                    else:
                        atoms[a_name] = {k:float(v) for k,v in line.items() if "boundary" not in k and k not in ['Dataset name', 'Atom name', 'Region name']}
        systems[sys] = {'atoms':atoms, 'regions':regions, "minmax_basins":minmax_basins}
    
    
    for sk, sv in systems.items():
        # map SGB regions to particular max/min basins
        for rk, rv in sv['regions'].items():
            def_var1 = rk.split(' from ')[1]
            minmax_basin_map = {}
            for cb1k, cb1v in rv.items():
                for cb2k, cb2v in sv['minmax_basins'].items():
                    found = False
                    def_var2 = cb2k.split(' from ')[1]
                    if def_var1 == def_var2 and cb2k[:cb2k.find(':')] == cb1k:
                        found = True
                        for vk, vv in cb1v.items():
                            found &= vk in cb2v and vv == cb2v[vk]
                    if found:
                        minmax_basin_map[cb1k] = cb2k
                        break
            if len(minmax_basin_map) != len(rv):
                print(f"Failed to associate {cb1k} in {rk} of {sk} with its corresponding condensed max or min basin!")
            systems[sk]['regions'][rk]['minmax_basin_map'] = minmax_basin_map
    
        # get min/max basins that don't take part in any special gradient bundles
        lone_minmax_basins = {}
        for ak in sv['atoms'].keys():
            lone_minmax_basins = {}
            for cbk in sv['minmax_basins'].keys():
                if cbk[:cbk.find(":")] == ak:
                    found = False
                    for rk, rv in sv['regions'].items():
                        found = (cbk in list(rv['minmax_basin_map'].values()))
                        if found:
                            break
                    if not found:
                        lone_minmax_basins[cbk] = True
            systems[sk]['atoms'][ak]['lone_minmax_basins'] = lone_minmax_basins
        
        # determine the sgbs composure of min/max or mixed
        sgb_types = {}
        for rk,rv in sv['regions'].items():
            ismax = all(["Max" in rv['minmax_basin_map'][ak] for ak in rv.keys() if "_map" not in ak])
            ismin = all(["Min" in rv['minmax_basin_map'][ak] for ak in rv.keys() if "_map" not in ak])
            if ismax:
                sgb_types[rk] = SGB_CONDENSED_BASIN_TYPE.MAX
            elif ismin:
                sgb_types[rk] = SGB_CONDENSED_BASIN_TYPE.MIN
            else:
                sgb_types[rk] = SGB_CONDENSED_BASIN_TYPE.MIXED
        systems[sk]['region_types'] = sgb_types
                
    
    
    ##################################
    # GET INITIAL SYSTEM, ATOM ASSOCIATIONS, AND SYSTEM NICKNAMES
    ################################## 
    
    # have the user confirm the association of atoms in the neutral (initial) system to those of the perturbed (final) systems
    # first they specify the neutral system
    str1 = '\n'.join([f"{i+1}. {s}" for i,s in enumerate(systems.keys())])
    initial_sys = False
    while not initial_sys:
        initial_sys = input("Select the initial, neutral system to which the other systems are being compared (enter a number)"+f"\n\n{str1}\n\nEnter a number: ")
        try:
            s_name = list(systems.keys())[int(initial_sys) - 1]
            initial_sys = s_name
        except:
            print(f"\n\nYou entered: {initial_sys}, which was not valid.\nLet's try that again...\n\n")
            initial_sys = False
    print(f"Selected initial system: {initial_sys}")
    
                        
                    
        
        
    
    # create map of atoms in the perturbed systems to atoms of the initial system, starting with a one-to-one assumption
    # store map in text file for the user to edit
    str1 = ''
    for k,v in systems.items():
        if k == initial_sys:
            str1 += f"[{k}] {k}\n"
            continue
        else:
            str1 += f"[{k}] {k} --> {initial_sys}\n"
            for a in v['atoms'].keys():
                str1 += f"{a} --> {a} --> {a}\n"
        str1 += "\n"
    
    
    file_name1 = f"{date_time_stamp}_AtomAssociations.txt"
    with open(file_name1, 'w') as f:
        f.write(str1)
    
    input(f"""

TIME TO ASSOCIATE ATOMS IN THE FINAL SYSTEM(S) WITH THEIR COUNTERPARTS IN THE INITIAL SYSTEM.

Press enter to open a file in which you'll define system nicknames [in brackets] and atom associations. It will open for you, but can be found at \"{os.path.join(os.getcwd(),file_name1)}\"

To allow for symmetry-degenerate atoms, you specify two associations, one internal to the system (i.e. an atom in the system pointing to an different atom in the same system), and one that points to an atom in the initial, unperturbed system.

In this file you can define which atoms in the initial/final systems map to which atoms. The triple association allows you to 'create' atoms that simply point to a different atom, thereby having a full list of atoms in a system where you perhaps only analyzed the symmetry-unique atoms.

Eg        C3 --> C1 --> C2         means that there will be an atom in the output called 'C3' that points to 'C1' in the specified system, that then is compared to 'C2' in the initial system.

example file before being edited:

    [cis-1-3-butadiene_EEF00125-y_SP] cis-1-3-butadiene_EEF00125-y_SP --> cis-1-3-butadiene_SP
    C1 --> C1 --> C1
    C2 --> C2 --> C2
    C3 --> C3 --> C3
    C4 --> C4 --> C4
    H1 --> H1 --> H1
    H2 --> H2 --> H2
    H3 --> H3 --> H3
    H4 --> H4 --> H4
    H5 --> H5 --> H5
    H6 --> H6 --> H6
    
    [cis-1-3-butadiene_EEF00125-z_SP] cis-1-3-butadiene_EEF00125-z_SP --> cis-1-3-butadiene_SP
    C1 --> C1 --> C1
    C2 --> C2 --> C2
    H4 --> H4 --> H4
    H5 --> H5 --> H5
    H6 --> H6 --> H6
    
    [cis-1-3-butadiene_SP] cis-1-3-butadiene_SP


same file after being edited:

    [EEF Y] cis-1-3-butadiene_EEF00125-y_SP --> cis-1-3-butadiene_SP
    C1 --> C1 --> C1
    C2 --> C2 --> C2
    C3 --> C3 --> C2
    C4 --> C4 --> C1
    H1 --> H1 --> H5
    H2 --> H2 --> H6
    H3 --> H3 --> H4
    H4 --> H4 --> H4
    H5 --> H5 --> H5
    H6 --> H6 --> H6
    
    [EEF Z] cis-1-3-butadiene_EEF00125-z_SP --> cis-1-3-butadiene_SP
    C1 --> C1 --> C1
    C2 --> C2 --> C2
    C3 --> C2 --> C2
    C4 --> C1 --> C1
    H1 --> H5 --> H5
    H2 --> H6 --> H6
    H3 --> H4 --> H4
    H4 --> H4 --> H4
    H5 --> H5 --> H5
    H6 --> H6 --> H6
    
    [NEF] cis-1-3-butadiene_SP



The atom association file for your systems will now open. Please edit it so that the atoms in each indicated system point to their correct counterparts in the initial system: {initial_sys}

Press enter to continue...\n\n""")
    
    os.system(f"open {file_name1}")
    
    input("Once you've verified the associations and made any necessary changes, save the file, return to this window and then press enter. (note that you can also copy-paste the contents of a previous atom associations file if you've already done this before for these systems)\n\nPress enter to continue, once you've made any necessary edits and saved the file...")
        
    num_errors = 1
    systems_copy = {k:v for k,v in systems.items()}
    while num_errors > 0:        
        # Read the file back in to get the associations
        with open(file_name1, 'r') as f:
            str2 = f.read().strip()
        
        try:
            num_errors = 0
            for sys in str2.split('\n\n'):
                lines = sys.split('\n')
                s_nickname = lines[0][lines[0].find("[")+1:lines[0].find("]")]
                s_name = lines[0][lines[0].find("]")+2:]
                if len(s_name.split(' --> ')) > 1:
                    s_name = s_name.split(' --> ')[0]
                else:
                    systems[initial_sys]['nickname'] = s_nickname
                    continue
                atom_map = {}
                internal_atom_map = {}
                for a in lines[1:]:
                    a123 = [ai.strip().replace(' ','') for ai in a.split(' --> ')]
                    if a123[2] not in systems[initial_sys]['atoms']:
                        print(f"Atom {a123[0]} (points to {a123[1]} in {s_name}) then points to {a123[2]}, which is not in {initial_sys}!")
                        num_errors += 1
                        continue
                    elif a123[1] not in systems[s_name]['atoms']:
                        print(f"Atom {a123[0]} points to {a123[1]} which is not in {s_name}!")
                        num_errors += 1
                        continue
                    
                    # process pointer atoms, updating minmax basins and creating new regions as necessary.
                    # only new logical bonds will be made, as it's feasible that a ring/cage could have e.g. three pairs of the same pair of atoms.
                    # with bonds, can assume 2 atoms per interaction, so if a bond already has two atoms and both are the targets of new pointer atoms,
                    # assume those two pointers constitute a separate bond that essentially points to that of the target atoms. 
                    if a123[0] not in systems[s_name]['atoms']:
                        systems[s_name]['atoms'][a123[0]] = systems[s_name]['atoms'][a123[1]]
                        minmax_basins = {}
                        # copy minmax basins for pointer atom
                        for cbk,cbv in systems[s_name]['minmax_basins'].items():
                            if a123[1] in cbk:
                                minmax_basins[cbk.replace(a123[1],a123[0])] = cbv
                        systems[s_name]['minmax_basins'].update(minmax_basins)
                        
                    atom_map[a123[1]] = a123[2] # maps existing atom to existing atom in initial system
                    atom_map[a123[0]] = a123[2] # maps new pointer to existing atom in initial system
                    internal_atom_map[a123[0]] = a123[1] # maps new pointer atom to existing atom
                    
                systems[s_name]['atom_map'] = atom_map
                
                region_map = {}
                for rk,rv in systems[s_name]['regions'].items():
                    for rki,rvi in systems[initial_sys]['regions'].items():
                        if all([int(systems[s_name]['atom_map'][a] in rvi) for a in rv.keys() if "_map" not in a]):
                            region_map[rk] = rki
                            break
                systems[s_name]['region_map'] = region_map
                systems[s_name]['nickname'] = s_nickname
                
                
                # update regions for pointer atom
                new_regions = {k:v for k,v in systems[s_name]['regions'].items()}
                for rk,rv in systems[s_name]['regions'].items():
                    for ak,av in internal_atom_map.items():
                        if ak != av and av in rv:
                            if "Bond " not in rk or len(rv) <= 2:
                                # singly bond-wedge occupied bond, or not a bond, so add pointer atom to this region
                                new_regions[rk]['minmax_basin_map'][ak] = rv['minmax_basin_map'][av].replace(av,ak)
                                new_regions[rk][ak] = rv[av]
                            elif "Bond " in rk and len(rv) == 3:
                                # full bond, so need to make a new one for this pointer atom.
                                # first check to see if a bond has already been made for this atom
                                # (i.e. a bond with the corresponding counterpart atom that was made in a previous iteration)
                                other_atom = [a for a in rv.keys() if a != av and "_map" not in a][0]
                                make_new_bond = True
                                if True or internal_atom_map[other_atom] == other_atom:
                                    for r2k,r2v in new_regions.items():
                                        if "Bond " in r2k and ' '.join(rk.split(" ")[2:]) == ' '.join(r2k.split(" ")[2:]) and len(r2v) == 2:
                                            for a2k in r2v.keys():
                                                if "_map" not in a2k and (internal_atom_map[a2k] == other_atom or internal_atom_map[a2k] == internal_atom_map[other_atom]):
                                                    new_regions[r2k][ak] = rv[av]
                                                    new_regions[r2k]['minmax_basin_map'][ak] = rv['minmax_basin_map'][av].replace(av,ak)
                                                    make_new_bond = False
                                                    break
                                        if not make_new_bond:
                                            break
                                                
                                    pass
                                if make_new_bond:
                                    def_var = rk.split(' from ')[1]
                                    bond_nums = [int(b.split(' ')[1]) for b in (systems[s_name]['regions'] | new_regions).keys() if "Bond " in b and def_var == b.split(' from ')[1]]
                                    if len(bond_nums) > 0:
                                        if min(bond_nums) > 1:
                                            bond_num = min(bond_nums) - 1
                                        else:
                                            bond_num = max(bond_nums) + 1
                                    else:
                                        bond_num = 1
                                    bond_name = f"Bond {bond_num} " + ' '.join(rk.split(" ")[2:])
                                        
                                    new_regions[bond_name] = {ak:rv[av], 'minmax_basin_map':{ak:rv['minmax_basin_map'][av].replace(av,ak)}}
                systems[s_name]['regions'] = new_regions
                systems[s_name]['internal_atom_map'] = internal_atom_map
                
                # update region map for new regions
                for r1k,r1v in systems[s_name]['regions'].items():
                    if "_map" not in r1k and all([internal_atom_map[a] != a for a in r1v.keys() if "_map" not in a]):
                        # found a new pointer region, so look for it's target region
                        for r2k,r2v in systems[s_name]['regions'].items():
                            if "_map" not in r2k and r2k != r1k and all([internal_atom_map[a] in r2v for a in r1v.keys() if "_map" not in a]):
                                # found target region
                                systems[s_name]['region_map'][r1k] = systems[s_name]['region_map'][r2k]
                            
                                    
                                    
            
            if num_errors:
                input(f"\n\nProblem(s) found with {num_errors} atom association(s). Please fix, resave the file, and press enter to check again.")
                systems = {k:v for k,v in systems_copy.items()}
                with open(file_name1, 'w') as f:
                    f.write(str1)
        except Exception as e:
            input(f"Problem reading the edited file: {str(e)}.\n\nLet's try that again... (ctrl-c to quit)")
            exc_type, value, exc_traceback = exc_info()
            print(f"{exc_type = }\n{value = }\n{exc_traceback = }")
            systems = {k:v for k,v in systems_copy.items()}
            with open(file_name1, 'w') as f:
                f.write(str1)
    
    systems = {sk:sv for sk,sv in systems.items() if 'atom_map' in sv or sk == initial_sys}
    
    ##################################
    # MAP SYSTEMS TO INITIAL SYSTEMS AND TO EACHOTHER
    ################################## 
    
    # map min/max basins to initial system based on correlation of integration values
    for sk,sv in systems.items():
        if sk == initial_sys:
            continue
        minmax_basin_correlations = {}
        for cb1k,cb1v in sv['minmax_basins'].items():
            cor_list = []
            a_name = cb1k[:cb1k.find(":")]
            def_var1 = cb1k.split(' from ')[1]
            for cb2k,cb2v in systems[initial_sys]['minmax_basins'].items():
                def_var2 = cb2k.split(' from ')[1]
                if def_var2 == def_var1 and sv['atom_map'][a_name] == cb2k[:cb2k.find(":")] and (all(["Max " in k for k in [cb1k,cb2k]]) or  all(["Min " in k for k in [cb1k,cb2k]])):
                    lvals = [v for k,v in cb1v.items() if "boundary" not in k and k in cb2v]
                    rvals = [v for k,v in cb2v.items() if "boundary" not in k and k in cb1v]
                    cor = np.corrcoef(np.array([lvals, rvals]))[0,1]
                    cor_list.append([cor,cb2k])
            minmax_basin_correlations[cb1k] = sorted(cor_list, key = lambda x: x[0], reverse = True)
        
        # loop over minmax_basin_correlations pairing basins with their closest matches
        is_added = {}
        minmax_basin_map = {}
        systems[sk]['minmax_basin_map_corrcoefs'] = {}
        for cb1k in minmax_basin_correlations.keys():
            a_name = cb1k[:cb1k.find(":")]
            if len(minmax_basin_correlations[cb1k]) < 1:
                continue
            def_var1 = cb1k.split(' from ')[1]
            while len(minmax_basin_correlations[cb1k]) > 1:
                do_break = True
                # check that no other minmax_basin has a better correlation with cb1k's best match
                for cb2k in minmax_basin_correlations.keys():
                    def_var2 = cb2k.split(' from ')[1]
#                     print(f"{cb1k = }\n{cb2k = }\n")
                    if def_var2 == def_var1 and cb1k != cb2k and a_name == cb2k[:cb2k.find(":")] and (all(["Max " in k for k in [cb1k,cb2k]]) or  all(["Min " in k for k in [cb1k,cb2k]])) and len(minmax_basin_correlations[cb2k]) > 0 and minmax_basin_correlations[cb2k][0][1] == minmax_basin_correlations[cb1k][0][1] and minmax_basin_correlations[cb2k][0][0] > minmax_basin_correlations[cb1k][0][0]:
                        del(minmax_basin_correlations[cb1k][0])
                        do_break = False
                        break
                if do_break:
                    break
            if len(minmax_basin_correlations[cb1k]) > 0:
                systems[sk]['minmax_basin_map_corrcoefs'][cb1k] = minmax_basin_correlations[cb1k][0][0]
                minmax_basin_map[cb1k] = minmax_basin_correlations[cb1k][0][1]
                is_added[minmax_basin_map[cb1k]] = True
        systems[sk]['minmax_basin_map'] = minmax_basin_map
        systems[sk]['initial_sys_unmapped_minmax_basins'] = {k:v for k,v in systems[initial_sys]['minmax_basins'].items() if k not in is_added}
    
    # reverse map of initial system minmax basins pointing to final system minmax basins
    reverse_minmax_basin_map = {}
    for cb1k,cb1v in systems[initial_sys]['minmax_basins'].items():
        reverse_minmax_basin_map[cb1k] = []
        for s2k,s2v in systems.items():
            if s2k != initial_sys:
                for cb2f,cb2i in s2v['minmax_basin_map'].items():
                    if cb1k == cb2i:
                        reverse_minmax_basin_map[cb1k].append([s2k,cb2f])
    
    
    
    ##################################
    # GET LIST OF VARIABLES TO INCLUDE
    ################################## 
    
    # now get list of variables to look at the same way
    # get list of variables present in all systems
    var_list = systems[initial_sys]['atoms'][list(systems[initial_sys]['atoms'].keys())[0]]
    for k,v in systems.items():
        var_list = {k1:v1 for k1,v1 in var_list.items() if k1 in v['atoms'][list(v['atoms'].keys())[0]] and ":" in k1}
    
    str1 = '\n'.join(list(var_list.keys()))
    
    file_name1 = f"{date_time_stamp}_Variables.txt"
    with open(file_name1, 'w') as f:
        f.write(str1)
    
    input(f"\n\nTIME TO SELECT AND ORDER VARIABLES TO BE COMPARED.\n\nPress enter to open the list of variable names present in all systems found that has been saved to \"{os.path.join(os.getcwd(),file_name1)}\"\n\nPlease order them how you'd like them to appear in the output and remove any that you don't want to analyze\n\n(If variables you wanted to look at are missing from the list, it means they weren't present in all systems so weren't included)")
    
    os.system(f"open {file_name1}")
    
    input("Once you've removed any unwanted variables, save the file, return to this window and then press enter. (note that you can also copy-paste the contents of a previous variable name file if you've already done this before for these systems)\n")
        
    num_errors = 1
    while num_errors > 0:        
        # Read the file back in to get the associations
        with open(file_name1, 'r') as f:
            str2 = f.read().strip()
        
        new_var_list = [s for s in str2.split('\n') if s.strip() != '']
        
        num_errors = 0
        for var in new_var_list:
            if var not in var_list:
                print(f"Variable {var} is invalid!")
        
        if num_errors:
            input(f"\n\nProblem(s) found with {num_errors} variable(s). Please fix, resave the file, and press enter to check again.")
    
    var_list = {k:True for k in new_var_list}
    
    
    ##################################
    # OUTPUT COMPARISONS
    ################################## 
    
    # Output comparisons in a couple forms:
    # 1. A file for each system comparison that includes all variables
    # 2. A file for each variable that includes all systems, where each system gets it's value/difference/percent change grouped together
    # 3. Same as (2) but with all values, all differences, and all percent changes grouped together
    
    # Calculating changes in each condensed basin for each variable, and reporting it as a total and as a percent change.
    # Layout of first type of output csv file will be similar to Table 1 of  https://doi.org/10.33774/chemrxiv-2021-9tjtw-v3
    # One file per final perturbed system:
    #     1. First show changes in atomic basins, 
    #     2. Section for each class of special gradient bundle (i.e. first bonds, then rings...), each with a followup section showing the non-bonded (or non-ringed) regions, 
    #     3. Section for atoms again, but this time with their complete list of max then min basins and the SGBs to which they belong, 
    
    # output 1
    dir_name = f"{date_time_stamp}_output"
    os.mkdir(dir_name)
    header_str1 = "Region," + ",".join([f"{v},,," for v in var_list])
    
    
    # get list of variables in use
    defining_var_list = {}
    for cbk in systems[initial_sys]['minmax_basins'].keys():
        v_name = cbk.split(" from ")[1]
        defining_var_list[v_name] = True
    
    for sk,sv in systems.items():
        if sk == initial_sys:
            continue
        header_str2 = "Atomic basin decomposition," + ",".join([f"{systems[initial_sys]['nickname']},{sv['nickname']},∆,%∆" for var in var_list.keys()])
        
        with open(os.path.join(dir_name,f"{sanitize(sk)} minus {sanitize(initial_sys)}.csv"), "w", encoding='utf_8_sig') as f:
            
            f.write(header_str1 + "\n" + header_str2 + "\n")
            
            # atomic basin decomposition
            def output_atomic_basin_decomposition():
                # write out a row for each atom in the final system
                for ak in sorted(list(sv['atoms'].keys())):
                    av = sv['atoms'][ak]
                    f.write(f"{ak},") 
                    for var in var_list.keys():
                        i_val = systems[initial_sys]['atoms'][sv['atom_map'][ak]][var]
                        f_val = av[var]
                        diff = f_val - i_val
                        percent_diff = (diff / i_val * 100.) if i_val != 0. else 0.
                        f.write(f"{i_val:.8f},{f_val:.8f},{diff:.8f},{percent_diff:.8f},")
                    f.write('\n')
                # the total line
                f.write("Total,")
                for var in var_list.keys():
                    i_tot = sum([systems[initial_sys]['atoms'][sv['atom_map'][ak]][var] for ak in sv['atoms'].keys()])
                    f_tot = sum([sv['atoms'][ak][var] for ak in sv['atoms'].keys()])
                    diff = f_tot - i_tot
                    percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                    f.write(f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},")
                f.write('\n')
                
            output_atomic_basin_decomposition()
            
            # special gradient bundle decompositions
            def output_decomposition(r_type):
                # r-type bundle decomposition (One of "Bond ", "Ring ", "Cage ") using bundles defined by one or more variables
                for def_var in defining_var_list.keys():
                
                    # write out a row for each bond bundle in the final system, with it's component bond wedges
                    var_itotals = {k:0. for k in var_list.keys()}
                    var_ftotals = {k:0. for k in var_list.keys()}
                    num_regions = 0
                    for rk in sorted(list(sv['regions'].keys()), reverse=True):
                        rv = sv['regions'][rk]
                        if def_var in rk and r_type in rk:
                            if num_regions == 0:
                                f.write(f"{r_type}bundle decomposition according to {def_var}\n")
                            num_regions += 1
                            # region info
                            r_name = f"{' — '.join([k for k in sorted(list(rv.keys())) if '_map' not in k])}" + f" {r_type.lower()}bundle"
                            f.write(f"{r_name},") 
                            
                            # region var totals
                            var_ivals = {k:0. for k in var_list.keys()}
                            var_fvals = {k:0. for k in var_list.keys()}
                            for var in var_list.keys():
                                for ak, av in rv.items():
                                    if not "_map" in ak:
                                        try:
                                            var_ivals[var] += systems[initial_sys]['minmax_basins'][sv['minmax_basin_map'][rv['minmax_basin_map'][ak]]][var]
                                            var_fvals[var] += av[var]
                                            var_itotals[var] += systems[initial_sys]['minmax_basins'][sv['minmax_basin_map'][rv['minmax_basin_map'][ak]]][var]
                                            var_ftotals[var] += av[var]
                                        except Exception as e:
                                            print(f"Exception fetching value for total {r_type} value for {var} for {ak} in {rk} in {sk} defined by {def_var}: {str(e)}")
                                            exc_type, value, exc_traceback = exc_info()
                                            print(f"{exc_type = }\n{value = }\n{exc_traceback = }")
                            for var in var_list.keys():
                                diff = var_fvals[var] - var_ivals[var]
                                percent_diff = (diff / var_ivals[var] * 100.) if var_ivals[var] != 0. else 0.
                                f.write(f"{var_ivals[var]:.8f},{var_fvals[var]:.8f},{diff:.8f},{percent_diff:.8f},")
                            f.write('\n')
                            
                            # constituent condensed basin values
                            for cb2k in sorted(list(rv.keys())):
                                cbv = rv[cb2k]
                                if not "_map" in cb2k:
                                    f.write(f"      ↳ {cb2k} {r_type.lower()}wedge,")
                                    for var in var_list.keys():
                                        try:
                                            i_val = systems[initial_sys]['minmax_basins'][sv['minmax_basin_map'][rv['minmax_basin_map'][cb2k]]][var]
                                            f_val = cbv[var]
                                            diff = f_val - i_val
                                            percent_diff = (diff / i_val * 100.) if i_val != 0. else 0.
                                            f.write(f"{i_val:.8f},{f_val:.8f},{diff:.8f},{percent_diff:.8f},")
                                        except Exception as e:
                                            print(f"Exception fetching value for bundle constituent min/max basin for {var} for {ak} in {rk} in {sk} defined by {def_var}: {str(e)}")
                                            exc_type, value, exc_traceback = exc_info()
                                            print(f"{exc_type = }\n{value = }\n{exc_traceback = }")
                                    f.write('\n')
                    
                    if num_regions == 0:
                        return
                                    
                    # any lone regions
                    # Because the max and min condensed basins both constitute a full partitioning of the system, can't just
                    # throw all the unused "lone" max/min basins in. Need to pick the type that completes whatever partitioning is being used.
                    # For bond bundle partitioning, it's _usually_ all max basins, so we can include all the non-bonded max basins to finish it off.
                    # Rather than assume, however, we'll check that the regions are all max or min condensed basins
                    ismin, ismax = [all([t == tt for r,t in sv['region_types'].items() if "Bond" in r]) for tt in [SGB_CONDENSED_BASIN_TYPE.MIN, SGB_CONDENSED_BASIN_TYPE.MAX]]
                    lone_regions = {}
                    if ismin:
                        for ak,av in sv['atoms'].items():
                            for cb2k in av['lone_minmax_basins'].keys():
                                if def_var in cb2k and "Min " in cb2k:
                                    lone_regions[cb2k] = sv['minmax_basins'][cb2k]
                    elif ismax:
                        for ak,av in sv['atoms'].items():
                            for cb2k in av['lone_minmax_basins'].keys():
                                if def_var in cb2k and "Max " in cb2k:
                                    lone_regions[cb2k] = sv['minmax_basins'][cb2k]
                    else:
                        f.write(f"\n{r_type.upper()}BUNDLE PARTITIONING INCLUDES CONTRIBUTIONS FROM BOTH MINIMUM AND MAXIMUM BASINS. THIS PREVENTS THE AUTOMATIC COMPLETION OF DECOMPOSITION WITH LONE REGIONS\n")
                    
                    if len(lone_regions) > 0:
                        unmapped = []
                        for lrk in sorted(list(lone_regions.keys())):
                            lrv = lone_regions[lrk]
                            if def_var in lrk and lrk in sv['minmax_basin_map']:
                                f.write(f"Lone: {lrk.replace(' from ' + def_var,'')} → {sv['minmax_basin_map'][lrk].replace(' from ' + def_var,'')} in {systems[initial_sys]['nickname']},")
                                for var in var_list.keys():
                                    i_tot = lrv[var]
                                    var_itotals[var] += i_tot
                                    f_tot = systems[initial_sys]['minmax_basins'][sv['minmax_basin_map'][lrk]][var]
                                    var_ftotals[var] += f_tot
                                    diff = f_tot - i_tot
                                    percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                                    f.write(f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},")
                                f.write('\n')
                            elif def_var in lrk:
                                unmapped.append(lrk)
                        
                        if len(unmapped) > 0:
                            for lrk in unmapped:
                                f.write(f"UNMAPPED lone: {lrk.replace(' from ' + def_var,'')},")
                                for var in var_list.keys():
                                    f_tot = lone_regions[lrk][var]
                                    var_ftotals[var] += f_tot
                                    f.write(f",{f_tot:.8f},,,")
                                f.write('\n')
                        
                        if len(sv['initial_sys_unmapped_minmax_basins']) > 0:
                            for lrk in sorted(list(sv['initial_sys_unmapped_minmax_basins'].keys())):
                                lrv = sv['initial_sys_unmapped_minmax_basins'][lrk]
                                if def_var in lrk and (ismax and "Max " in lrk) or (ismin and "Min " in lrk):
                                    f.write(f"UNMAPPED lone: {lrk.replace(' from ' + def_var,'')} in {systems[initial_sys]['nickname']},")
                                    for var in var_list.keys():
                                        i_tot = lrv[var]
                                        var_itotals[var] += i_tot
                                        f.write(f"{i_tot:.8f},,,,")
                                    f.write('\n')
                                
                    
                            
                            
                    # the total line
                    f.write("Total,")
                    for var in var_list.keys():
                        i_tot = var_itotals[var]
                        f_tot = var_ftotals[var]
                        diff = f_tot - i_tot
                        percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                        f.write(f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},")
                    f.write('\n')
            
            output_decomposition("Bond ")
            output_decomposition("Ring ")
            output_decomposition("Cage ")
            
            # max and min basin decomposition
            def output_minmax_basin_decompositions():
                for def_var in defining_var_list.keys():
                    for m in ['Max ',"Min "]:
                        f.write(f"\n{m}basin decomposition according to {def_var}\n")
                        unmapped = []
                        var_itotals = {k:0. for k in var_list.keys()}
                        var_ftotals = {k:0. for k in var_list.keys()}
                        for cb1k in sorted(list(sv['minmax_basins'].keys())):
                            cb1v = sv['minmax_basins'][cb1k]
                            if def_var not in cb1k or m not in cb1k:
                                continue
                            if cb1k in sv['minmax_basin_map']:
                                f.write(f"{cb1k.replace(' from ' + def_var,'')} → {sv['minmax_basin_map'][cb1k].replace(' from ' + def_var,'')} in {systems[initial_sys]['nickname']} ({sv['minmax_basin_map_corrcoefs'][cb1k]:.8f}),")
                                for var in var_list.keys():
                                    i_tot = cb1v[var]
                                    var_itotals[var] += i_tot
                                    f_tot = systems[initial_sys]['minmax_basins'][sv['minmax_basin_map'][cb1k]][var]
                                    var_ftotals[var] += f_tot
                                    diff = f_tot - i_tot
                                    percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                                    f.write(f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},")
                                f.write('\n')
                            else:
                                unmapped.append(cb1k)
                        
                        if len(unmapped) > 0:
                            for cb1k in unmapped:
                                if m not in cb1k:
                                    continue
                                f.write(f"UNMAPPED: {cb1k.replace(' from ' + def_var,'')},")
                                for var in var_list.keys():
                                    f_tot = sv['minmax_basins'][cb1k][var]
                                    var_ftotals[var] += f_tot
                                    f.write(f",{f_tot:.8f},,,")
                                f.write('\n')
                        
                        if len(sv['initial_sys_unmapped_minmax_basins']) > 0:
                            for cb1k in sorted(list(sv['initial_sys_unmapped_minmax_basins'].keys())):
                                cb1v = sv['initial_sys_unmapped_minmax_basins'][cb1k]
                                if def_var in cb1k and (m in cb1k):
                                    f.write(f"UNMAPPED: {cb1k.replace(' from ' + def_var,'')} in {systems[initial_sys]['nickname']},")
                                    for var in var_list.keys():
                                        i_tot = cb1v[var]
                                        var_itotals[var] += i_tot
                                        f.write(f"{i_tot:.8f},,,,")
                                    f.write('\n')
                        
                        # the total line
                        f.write("Total,")
                        for var in var_list.keys():
                            i_tot = var_itotals[var]
                            f_tot = var_ftotals[var]
                            diff = f_tot - i_tot
                            percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                            f.write(f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},")
                                    
            output_minmax_basin_decompositions()
                        
    
    # output 2
    # finish this later; 
    max_num_reverse_minmax_basin_mappings = max([len(v) for v in reverse_minmax_basin_map.values()])
    
    print(f"Finished! You'll find your comparison files in {dir_name} next to the input files")
    
if __name__ == "__main__":
#     try:
    inpath = argv[1] if len(argv) > 1 else "/Volumes/twilson/Tecplot/StorageWorkspace/2021_QTAIM-book-chapter/a/integration data 1/cis-1-3-butadiene"
    os.chdir(inpath)
    main()
#     except Exception as e:
#         print(f"Something went wrong. To use, run:\n\npython3 path/to/gbapost.py path/to/folder/of/gba/results\n\neg. python3 ~/Desktop/gpapost.py ~/Desktop/results\n\nerror: {e}")
#         exc_type, value, exc_traceback = exc_info()
#         print(f"{exc_type = }\n{value = }\n{exc_traceback = }")