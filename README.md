# Electron phase plate simulation software to see and assess performance of an electron phase plate on different specimens.

## How to use TEM simulator with electron phase plate implementation

The following steps should be followed when
trying to generate images using the simulation
program.

1. Correctly install all the package dependencies which are:
   * Numpy 1.26.2
   * Matplotlib 3.8.2
   * mrcfile 1.5.0
   * Scipy 1.12.0
   * Jupyter (Latest Version)
   EMAN 2.99.47 must also be installed to rotate the molecule and generate the detector noise.
2. Run the MATLAB script from Shang and Sigworth (https://www.sciencedirect.com/science/article/pii/S1047847712001244?via%3Dihub) after acquiring the appropriate pdb file to acquire the electron density map of the macromolecule. Alternatively use the `pdb2mrc.py` program from EMAN2 to perform the same procedure. Note that the boxsize of the mrcfile must be 256 × 256 × 256.
3. Modify the parameters accordingly and run the `Generate_Rotations.ipynb` file in a jupyter notebook and acquire the different rotations of the macromolecule. Make sure that the program is run from the top to the bottom as the run order matters for correct execution of the program. It is also crucial that the names of the rotated molecules are chosen in this specific way: `{original molecule name}_rotated_{i}` where the original molecule name is the name of the original mrc file acquired from the pdb file and i is the number of the rotation. This means that for each rotation generated i should be incremented by one. The easiest way to run this correctly would be to follow the example already present in the jupyter notebook file.
4. Change the `filename` variable in the `multislice.py` file to the name of the mrc file that is the original file acquired directly from the pdb file.
5. Change the constant values for the number of beam electrons to the number of potential maps to be acquired and then run the `create_Potential_Maps()` function.
6. After the potential maps have been acquired simply run the function `generate_all_projections()` which takes the number of rotations of the molecule acquired, the file name of the original mrc file (without the .mrc extension), number of projections to be acquired, the vector/list containing all of the defocus values for each projection to be acquired and the relative noise level as input arguments. Note that the number of defocus values in the list and the number of projections to be acquired must be the same.
7. The next step is to use the `e2proc2d.py` function in EMAN2 to convert all of the mrc files to hdf files. After this conversion simply run the `Generate_Detector_Noise.ipynb` file from top to bottom and all of the images to be used for 3D reconstruction will then be present in a new folder called images with noise.
8. The final step is to follow the general pipeline of EMAN2 for Single Particle Analysis.

## Contact Details
If any queries show up contact us at: \
smyan@kth.se \
hugosv@kth.se \
antonj2@kth.se 
