# Gradient bundle analysis system comparison tool

#### This document accompanies the example input and output for cis-1,3-butadiene analyzed with the `gbapost.py` tool.

---

This tool is for comparing chemical systems analyzed with gradient bundle analysis (GBA). The paradigm is simple; compute the differences of integrated totals of gradient bundle condensed variables in regions of a reference “initial” system to counterpart regions in a perturbed “final” system.   

You can compare atom/bond/ring/cage/molecule bundle energies/volumes/populations/etc as you alter the external potential by e.g. moving atoms, swapping atom types, adding solvation or electric fields, or hypothetically any other way. Clearly you need to believe that the systems you’re comparing constitute a valid comparison. 

We’ll use cis-1,3-butadiene as a quick example.

---

## 1) Setting up the system
The neutral molecule has a mirror plane (XZ), so only the atoms on the right need be analyzed with GBA. When an external electric field in the Y direction, the mirror symmetry is broken, and all atoms need to be analyzed.



We can still perform an analysis on the full molecule by specifying a logical mapping of atoms in the final systems to atoms of the initial system. This also recovers the special gradient bundles present between pointer atoms. For example, in this system, only one C–C bond is fully present (C1–C2), but we will also recover the C2–C3 and C3–C4 bonds, and the rest.

After running GBA on the atoms shown, open the GBA dialog “results” tab, select the condensed charge density “INS: Electron Charge Density” in the variable list, and click “find special gradient bundles”. You can do this multiple times to generate the special gradient bundle decompositions based on several condensed variables. Once this completes, select ​*MTG\_Utilities → Export gradient bundle integration data.*​ Select **at least** atomic basins, special gradient bundles, and max/min condensed basins for export.





---

## 2) Using the gbapost.py tool
1. Point the tool to a folder containing exported csv files from GBA for the systems being compared (you need to include at least atomic basins, special gradient bundles, and min/max condensed basins in the export!). There can be one initial system and one or more final systems. Each must have the same set of files for atomic basins, special gradient bundles, etc.
2. 
	- The example includes 5 systems; the neutral molecule, and four others with different oriented external electric fields. Each will be compared to the initial neutral systemIn terminal, run `python3 path/to/gbapost.py path/to/results/folder` (eg `python3 /Users/user/Desktop/gbapost.py /Users/user/Desktop/results` ). The tool will read in all the csv files and associate regions in one file with those in the rest.
3. Specify which system is to be the initial system
4. The tool prepares a guess input file to specify atomic associations between files. This file needs to be edited and saved to specify the actual atom associations.
	- Here's the partial contents of the file originally:

```
...
```

 ```
[cis-1-3-butadiene\_EEF00125-y\_SP] cis-1-3-butadiene\_EEF00125-y\_SP --> cis-1-3-butadiene\_SP C1 --> C1 --> C1 C2 --> C2 --> C2 C3 --> C3 --> C3 C4 --> C4 --> C4 H1 --> H1 --> H1 H2 --> H2 --> H2 H3 --> H3 --> H3 H4 --> H4 --> H4 H5 --> H5 --> H5 H6 --> H6 --> H6
```

 ```
[cis-1-3-butadiene\_EEF00125-z\_SP] cis-1-3-butadiene\_EEF00125-z\_SP --> cis-1-3-butadiene\_SP C1 --> C1 --> C1 C2 --> C2 --> C2 H4 --> H4 --> H4 H5 --> H5 --> H5 H6 --> H6 --> H6
```

 ```
[cis-1-3-butadiene\_SP] cis-1-3-butadiene\_SP
```


	- The file has a section for each final system, with a heading showing the system nickname, its "full name", and the full name of the initial system to which it is being compared, followed by a line for each atom present in the data for the final system.
	- Each atom line has three columns separated by " --> "
		- The first column indicates a "pointer" atom that exists in the *real* system that you want to appear in the output. It doesn't need to be present in the actual data; instead it points to the atom that it is symmetrically equivalent to in the second column
		- The second column indicates an atom that is present in the data for the listed final system. This atom needs to be in the data, and can be pointed to by one or more pointer atoms. It then points to an atom in the initial system in the third column
		- The third column indicates an atom present in the initial system.
		- If an atom in the second column isn't in the final system, or an atom in the thrid column isn't in the initial system, you will be asked to fix it and try again
	- The square-bracketed beginning of each line is a nickname you can give to the system, to make the output csv files easier to read.
	- Note that the atoms are not shown for the initial system, "cis-1-3-butadiene\_SP," and that different final systems have different numbers of atoms. They all represent the same molecule, but some atoms were left out thanks to symmetry. Here's the same part of the file, but with changes applied:

```
...
```

 ```
[EEF Y] cis-1-3-butadiene\_EEF00125-y\_SP --> cis-1-3-butadiene\_SP C1 --> C1 --> C1 C2 --> C2 --> C2 C3 --> C3 --> C2 C4 --> C4 --> C1 H1 --> H1 --> H5 H2 --> H2 --> H6 H3 --> H3 --> H4 H4 --> H4 --> H4 H5 --> H5 --> H5 H6 --> H6 --> H6
```

 ```
[EEF Z] cis-1-3-butadiene\_EEF00125-z\_SP --> cis-1-3-butadiene\_SP C1 --> C1 --> C1 C2 --> C2 --> C2 C3 --> C2 --> C2 C4 --> C1 --> C1 H1 --> H5 --> H5 H2 --> H6 --> H6 H3 --> H4 --> H4 H4 --> H4 --> H4 H5 --> H5 --> H5 H6 --> H6 --> H6
```

 ```
[NEF] cis-1-3-butadiene\_SP
```


	- See that I've given the systems nicknames (EEF for external electric field, and NEF for no electric field).
	- Also see that I've added extra lines in the "EEF Z" system, so that the output will have the FULL list of atoms for all systems, even though not all atoms were explicitly analyze
	- Having made these changes and saved the file, I can press enter to prompt gbapost.py to continue
5. The atom associations will be read in and checked for logical consistency. You'll be prompted if there are errors.
6. Now you do a similar process, but simpler, to select which condensed variables should be included in the output. Simply delete the lines for the variables you don't want, save, and press enter to prompt gbapost.py to continue
7. Now you should find an output folder next to the input files with one csv file for each final used. It will contain an atomic basin decomposition comparison, all special gradient bundle decompositions present in the system, and all max/min basin decompositions present.
	- For example, the bond/ring/cage bundles in a system can be defined according to the condensed density, the condensed kinetic energy density, or the condensed volume (among others). This tool will give the bond decomposition, the ring decomposition, and the cage decomposition, and according to each defining variable. So you get the bonds according to the condensed density, and another decomposition with the bonds according to the condensed volume, etc.

---

## 3) Output data
Here’s a simplified partial example of the output that would be produced using the example above

| Region | I: Electron Density SCF |  |  |  |
| --- | --- | --- | --- | --- |
| Atomic basin decomposition | NEF | EEF Y | ∆ | %∆ |
| C1 | 6.037927 | 5.9851832 | -0.0527438 | -0.87354153 |
| C2 | 6.0139573 | 6.0240909 | 0.0101336 | 0.16850136 |
| C3 | 6.0139573 | 6.0020398 | -0.0119175 | -0.19816403 |
| C4 | 6.037927 | 6.0932506 | 0.0553236 | 0.91626812 |
| H1 | 0.93725702 | 1.0190532 | 0.08179618 | 8.72718777 |
| H2 | 0.94851679 | 0.94875738 | 0.00024059 | 0.02536486 |
| H3 | 0.94551411 | 0.96974158 | 0.02422747 | 2.56235943 |
| H4 | 0.94551411 | 0.918734 | -0.02678011 | -2.83233319 |
| H5 | 0.93725702 | 0.85828249 | -0.07897453 | -8.42613374 |
| H6 | 0.94851679 | 0.94468449 | -0.0038323 | -0.4040308 |
| Total | 29.76634444 | 29.76381764 | -0.0025268 | -0.00848878 |
|  |  |  |  |  |
| Bond bundle decomposition according to INS: Electron Density SCF |  |  |  |  |
| C1 — H5 bond bundle | 2.75483222 | 2.80128849 | 0.04645627 | 1.68635569 |
| ↳ C1 bond wedge | 1.8175752 | 1.943006 | 0.1254308 | 6.90099645 |
| ↳ H5 bond wedge | 0.93725702 | 0.85828249 | -0.07897453 | -8.42613374 |
| C1 — H6 bond bundle | 2.76566989 | 2.75096259 | -0.0147073 | -0.53178075 |
| ↳ C1 bond wedge | 1.8171531 | 1.8062781 | -0.010875 | -0.59846361 |
| ↳ H6 bond wedge | 0.94851679 | 0.94468449 | -0.0038323 | -0.4040308 |
| … |  |  |  |  |
| Total | 29.76634424 | 29.76381764 | -0.0025266 | -0.00848811 |
| Max basin decomposition according to INS: Electron Density SCF |  |  |  |  |
| C1: Max (node 6104) → C1: Max (node 6148) in NEF (0.99986561) | 2.2358992 | 2.4031987 | 0.1672995 | 7.48242586 |
| C1: Max (node 6126) → C1: Max (node 6168) in NEF (0.99986342) | 1.8062781 | 1.8171531 | 0.010875 | 0.60206676 |
| … |  |  |  |  |
| Total | 29.76381764 | 29.76634424 | 0.0025266 | 0.00848883 |
|  |  |  |  |  |
| Min basin decomposition according to INS: Electron Density SCF |  |  |  |  |
| C1: Min (node 2120) → C1: Min (node 2804) in NEF (0.99977556) | 2.0159573 | 1.7902684 | -0.2256889 | -11.19512303 |
| C1: Min (node 3373) → C1: Min (node 3390) in NEF (0.99844260) | 0.83587636 | 1.1337787 | 0.29790234 | 35.63952209 |
| … |  |  |  |  |
| Total | 29.76381776 | 29.76634444 | 0.00252668 | 0.0084891 |

- A csv file with data like this will be produced for each final system that is compared to the initial system. 
- It will contain
	- an atomic basin decomposition, and
	- for each variable that was used in GBA to find special gradient bundles,
		- a bond/ring/cage bundle decomposition,
		- a condensed maximum basin decomposition, and 
		- a condensed minimum basin decomposition.
			- The max/min decompositions also shows which max/min basins were compared between the initial/final systems. 
			- The parenthetical number shown at the right of the first column in the min/max basin decomposition sections is the correlation coefficient (essentially r-squared) between the compared min/max basins, computed using their respective set of integrated properties.
			- Any unmapped max/min basins indicate a different *number* of max/min basins between the systems and provides a starting place to look at in case results don’t make sense. Typically if this occurs, the unmapped regions represent a very small amount of the integrated properties.
- Note how the system nicknames I specified in the atom association step are used extensively to make the output more readable and require less touching up in order to inspect/publish

Thank you, and please reach out to Tim Wilson at twilson@mines.edu if you have any questions!
