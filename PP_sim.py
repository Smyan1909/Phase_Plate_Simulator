import random
import math

import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import multislice as mt
import scipy.spatial.distance as distance
import csv
import os
import glob


pp_electrons = 200
beam_electrons = 20

#time_step = 1e-13

# constants
k = 8.988 * 10**9
pp_electron_velocity = 18.7e6
beam_electron_velocity = 265e6
m_e = 9.1093837e-31
e = 1.602 * 10 ** -19
c = 299792458  # speed of light
x_range = 0.05e-3
y_range = 0.05e-3
z_range = 2e-3

my0 = 1.2566370614*10**-6

angstrom = 1e-10
voxelsize = 1  # Ångström
# grid_size = 256  # 256x256 pixel grid for image

focal_length = 4e-3

class Electron:
    def __init__(self, charge=None, position=None, velocity=None, acceleration=None, mass=m_e, keV=None):
        self.charge = charge
        self.keV = keV
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.distance_matrix = None
        self.colomb_force_matrix = None
        self.magnetic_force_matrix = None
        self.lorentz_force_matrix = None
        self.net_force = None  # Added attribute for net force
        self.mass = mass

    def rk4_integrator(self, time_step, all_electrons):
        """
        Performs one step of RK4
        :param time_step: The time step of one RK4 step
        :param all_electrons: All the other electrons that induce forces on the current electron
        :return: Nothing
        """
        dt = time_step

        k1_v = dt*(self.total_force(all_electrons, self.position, self.velocity)/m_e)
        k1_x = dt * self.velocity

        k2_v = dt*(self.total_force(all_electrons, self.position + (k1_x/2), self.velocity + (k1_v/2))/m_e)
        k2_x = dt*(self.velocity + (k1_v/2))

        k3_v = dt*(self.total_force(all_electrons, self.position + (k2_x/2), self.velocity + (k2_v/2))/m_e)
        k3_x = dt*(self.velocity + (k2_v/2))

        k4_v = dt*(self.total_force(all_electrons, self.position + k3_x, self.velocity + k3_v)/m_e)
        k4_x = dt*(self.velocity + k3_v)

        self.velocity += (1/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        self.position += (1/6)*(k1_x + 2*k2_x + 2*k3_x + k4_x)

        if self.position[0] > x_range:
            self.position = np.array([np.random.normal((0.5e-6 + 0.25e-6) - x_range, 0.25e-6), 1e-6 * np.random.normal(0, 0.5),
                             1e-6 * np.random.normal(0, 0.5)])
            self.velocity = np.array([pp_electron_velocity, 0, 0])

    def Euler(self, time_step, all_electrons):
        """
        Performs one step of Eulers method
        :param time_step: The time step of one Euler step
        :param all_electrons: All the other electrons that induce forces on the current electron
        :return: Nothing
        """
        self.velocity += time_step*(self.total_force(all_electrons, self.position, self.velocity)/m_e)
        self.position += time_step*self.velocity

    def colomb_force(self, all_electrons, x):
        """
        Calculates the Colomb forces that all electrons have on one electron
        :param all_electrons: All the other electrons that induce forces on the current electron
        :param x: Position of the current electron
        :return: Sum of all Colomb forces on the current electron
        """
        colomb_forces = np.zeros((len(all_electrons), 3))  # Initialize forces array
        for i, other in enumerate(all_electrons):
            if other != self:
                rx = x[0] - other.position[0]
                ry = x[1] - other.position[1]
                rz = x[2] - other.position[2]

                distance = math.sqrt(rx**2 + ry**2 + rz**2)
                unit_vector = np.array([rx, ry, rz]) / distance

                # Calculate Coulomb force vector
                F_c = k * (self.charge * other.charge) / (distance**2)
                colomb_force_vector = unit_vector * F_c
                colomb_forces[i] = colomb_force_vector

        self.colomb_force_matrix = colomb_forces

        return np.sum(self.colomb_force_matrix, axis=0)

    def keV_to_ms (self):
        """
        Calculates the electron velocity from the electron energy
        :return: The velocity of the electron
        """
        return math.sqrt(2 * self.keV *1000*e/ m_e)

    def magnetic_force(self, all_electrons, x, v):
        """
         Biot–Savart law
        :param all_electrons: All the other electrons that induce forces on the current electron
        :param x: Position of the current electron
        :param v: The current electron velocity
        :return: The sum of the magnetic forces on the current electron
        """
        magnetic_forces = np.zeros((len(all_electrons), 3))
        for i, other in enumerate(all_electrons):
            if other != self:
                rx = x[0] - other.position[0]
                ry = x[1] - other.position[1]
                rz = x[2] - other.position[2]

                distance = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
                unit_vector = np.array([rx, ry, rz]) / distance

                cross_product = np.cross(v, unit_vector)

                b_factor = (my0*self.charge) / (4 * np.pi * distance ** 2)
                B = b_factor * cross_product
                F_b = self.charge*np.cross(v, B)
                magnetic_forces[i] = F_b

        self.magnetic_force_matrix = magnetic_forces

        return np.sum(self.magnetic_force_matrix, axis=0)


    def total_force(self, all_electrons, x, v):
        """
        Calculates the total forces acting on the current electron
        :param all_electrons: All the other electrons that induce forces on the current electron
        :param x: Position of the current electron
        :param v: The current electron velocity
        :return: The sum of Colomb forces and magnetic forces on the current electron
        """
        return self.colomb_force(all_electrons=all_electrons, x=x) + self.magnetic_force(all_electrons=all_electrons, x=x, v=v)

def calculate_potential(elec_array, step_size, start_range, stop_range):
    """
    Calculates the potential contribution from all electrons in the defined volume
    :param elec_array: Array containing the electrons
    :param step_size: The size of each step
    :param start_range: Starting point of the defined volume
    :param stop_range: Ending point of the defined volume
    :return: The potential in the defined volume and slice thickness
    """
    x_ = np.arange(start_range, stop_range+step_size, step_size)
    y_ = np.arange(start_range, stop_range+step_size, step_size)
    #z_ = np.arange(start_range, stop_range+step_size, step_size)
    z_ = np.linspace(-z_range, z_range, 400)
    x, y, z = np.meshgrid(x_, y_, z_, sparse=True)

    Vp = np.zeros((len(x_), len(y_), len(z_)))
    dz = z_[1]-z_[0]

    for electron in elec_array:

        elec_pos = np.array(electron.position)

        dist = np.sqrt((x - elec_pos[0]) ** 2 + (y - elec_pos[1]) ** 2 + (z - elec_pos[2]) ** 2)

        Vp += k * (electron.charge / dist)


    return Vp, dz


def calculate_potential_slice(elec_array, step_size, start_range, stop_range, z_val):
    """
    Calculates the potential contribution from all electrons in a slice of the defined volume
    :param elec_array: Array containing the electrons
    :param step_size: The size of each step
    :param start_range: Starting point of the slice
    :param stop_range: Ending point of the slice
    :param z_val: The z position of the slice
    :return: The potential of the slice
    """
    x_ = np.arange(start_range, stop_range + step_size, step_size)
    y_ = np.arange(start_range, stop_range + step_size, step_size)

    x, y = np.meshgrid(x_, y_, sparse=True)

    V_slice = np.zeros((len(x_), len(y_)))

    for electron in elec_array:

        elec_pos = np.array(electron.position)

        dist = np.sqrt((x - elec_pos[0])**2 + (y-elec_pos[1])**2 + (z_val - elec_pos[2])**2)

        V_slice += k*(electron.charge/dist)

    return V_slice
def calculate_potential_fast(elec_array, step_size):
    """
    A computationally faster version of the calculate_potential function
    :param elec_array: Array containing the electrons
    :param step_size: The size of each step
    :return: The potential in the defined volume and slice thickness
    """
    # Generate grid points
    x_ = np.linspace(-x_range, x_range, 100)
    y_ = np.linspace(-y_range, y_range, 100)
    z_ = np.linspace(-z_range, z_range, 100)
    x, y, z = np.meshgrid(x_, y_, z_, sparse=True)

    # Flatten the grid arrays for distance calculations
    grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Initialize the potential array
    Vp = np.zeros(grid_points.shape[0])

    # Calculate distances and update potentials for each electron
    for electron in elec_array:
        elec_pos = np.array([electron.position])  # Make it 2D for cdist compatibility
        # Calculate distances from this electron to all grid points
        dist = distance.cdist(grid_points, elec_pos).flatten()
        print(dist)
        # Update the potential array - avoid division by zero
        Vp += k*(-e/dist)

    # Reshape Vp to the original grid shape
    Vp = Vp.reshape(x.shape)

    # Calculate dz for return, assuming uniform spacing
    dz = z_[1] - z_[0]

    return Vp, dz

def update(num, all_electrons, dt, ax):
    """
    An update function to perform one frame update in an animation
    :param num:
    :param all_electrons: All electrons that are included in the animation
    :param dt: Time step
    :param ax: The figure object to animate
    """
    ax.set_xlim([-x_range, x_range])
    ax.set_ylim([-y_range, y_range])
    ax.set_zlim([-z_range, z_range])
    for electron in all_electrons:
        electron.rk4_integrator(all_electrons=all_electrons, time_step=dt)

    ax.clear()
    for electron in all_electrons:
        x = electron.position[0]
        y = electron.position[1]
        z = electron.position[2]
        #color = "blue" if electron in electron_array_pp else "red"
        color = "blue"
        ax.scatter(x, y, z, c=color)



# Räkna om nån beam electron slungas iväg z>0: se fulkod main
beam_electron_index = 0
frames_passed = 0
beam_positions = []


def tester_1():
    # Create an array of Electron objects with random initial positions
    electron_array_pp = [Electron(charge=e, keV=20, position=np.array([np.random.normal((i * 0.5e-6 + 0.25e-6)-x_range, 0.25e-6),
                                                                       1e-6 * np.random.normal(0, 0.5),
                                                                       1e-6 * np.random.normal(0, 0.5)]),
                                  velocity=np.array([pp_electron_velocity, 0, 0])) for i in range(pp_electrons)]

    electron_array_beam = [Electron(charge=e, velocity=np.array([0, 0, -1 * beam_electron_velocity]),
                                    position=[random.uniform(-x_range, x_range), 0, z_range]) for _ in
                           range(beam_electrons)]

    all_electrons = electron_array_pp + electron_array_beam


    # 3D plot of electron positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Extract x, y, z coordinates from each electron's position for pp_electrons
    """x_coords_pp = [electron.position[0] for electron in electron_array_pp]
    y_coords_pp = [electron.position[1] for electron in electron_array_pp]
    z_coords_pp = [electron.position[2] for electron in electron_array_pp]"""

    """
    # Extract x, y, z coordinates from each electron's position for beam_electrons
    x_coords_beam = [electron.position[0] for electron in electron_array_beam]
    y_coords_beam = [electron.position[1] for electron in electron_array_beam]
    z_coords_beam = [electron.position[2] for electron in electron_array_beam]
    """

    # Plotting electrons from pp_electrons
    #ax.scatter(x_coords_pp, y_coords_pp, z_coords_pp, c='b', marker='o', label='pp_electrons')

    """
    # Plotting electrons from beam_electrons
    ax.scatter(x_coords_beam, y_coords_beam, z_coords_beam, c='r', marker='s', label='beam_electrons')
    """

    #V, dz = calculate_potential(electron_array_pp,)

    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(111)
    #ax2.imshow(np.sum(V, axis=2) * dz)

    dt = 1e-13

    ani = FuncAnimation(fig, update, frames=range(200), fargs=(electron_array_pp, dt, ax))

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    # ax.legend()

    # Add a legend
    # ax.legend()

    # Display the plot
    plt.show()

def pp_stationary():
    electron_array_pp = [Electron(charge=-e, keV=20, position=np.array([np.random.normal((i * 0.5e-6 + 0.25e-6)-x_range, 0.25e-6),
                                                                       1e-6 * np.random.normal(0, 0.5),
                                                                       1e-6 * np.random.normal(0, 0.5)]),
                                  velocity=np.array([0, 0, 0])) for i in range(pp_electrons)]



    potential_calc_size, start_range, end_range = mt.freq_analysis()


    print("Generating Potential for phase plate ...")
    start_ppV_time = time.time()
    V, dz = calculate_potential(electron_array_pp, potential_calc_size, start_range, end_range)

    proj_V = np.sum(V, axis=2)*dz
    proj_V = proj_V - np.min(proj_V)
    end_ppV_time = time.time()
    print(f"Phase Plate potential calculated! (Time: {end_ppV_time-start_ppV_time}s)")

    x_vals, y_vals = mt.generate_grid(mt.pots)

    print("Performing Multislice ... ")
    start_mt_time = time.time()
    psi = mt.multislice(x_vals, y_vals, 200)
    end_mt_time = time.time()
    print(f"Multislice Complete! (Time: {end_mt_time-start_mt_time}s)")

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x_vals), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y_vals), d=(voxelsize * angstrom)))
    k_four = np.sqrt(kx ** 2 + ky ** 2)


    H_0 = mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, 0, 1)

    Im = np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(psi))*np.exp(-1j*mt.sigma_e*proj_V))

    ideal_image = mt.ideal_image()

    frc1, r_vals1 = fourier_ring_correlation(np.abs(np.fft.ifft2(Im*H_0))**2, ideal_image)

    frc2, r_vals2 = fourier_ring_correlation(np.abs(np.fft.ifft2(np.fft.fft2(psi)*mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, 82e-9, 1)))**2, ideal_image)

    radius_vals = np.linspace(0, 4.95, num=len(frc1))

    print("PSNR pp+lens = ", PSNR(ideal_image, np.abs(np.fft.ifft2(Im*H_0))**2))
    print("PSNR Scherzer = ", PSNR(ideal_image, np.abs(np.fft.ifft2(np.fft.fft2(psi)*mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, 82e-9, 1)))**2))

    plt.figure(1)
    plt.imshow(np.abs(np.fft.ifft2(Im))**2, cmap="gray")
    plt.xlabel("x [Å]")
    plt.ylabel("y [Å]")

    plt.figure(2)
    plt.imshow(np.abs(np.fft.ifft2(np.fft.fft2(psi)*mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, 82e-9, 1)))**2, cmap="gray")
    plt.xlabel("x [Å]")
    plt.ylabel("y [Å]")

    plt.figure(3)
    plt.imshow(np.abs(np.fft.ifft2(Im*H_0))**2, cmap="gray")
    plt.xlabel("x [Å]")
    plt.ylabel("y [Å]")

    plt.figure(4)
    plt.imshow(proj_V, extent=(-50, 50, -50, 50))
    plt.colorbar()
    plt.xlabel(r"x [$\mu m$]")
    plt.ylabel(r"y [$\mu m$]")

    plt.figure(5)
    plt.imshow(np.sin(-proj_V*mt.sigma_e), cmap="gray")

    plt.figure(6)
    plt.imshow(np.sin(-proj_V*mt.sigma_e + np.fft.fftshift(mt.lens_abber_func(k_four, mt.wavelength, 2e-3, 0))), cmap="gray")

    plt.figure(7)
    plt.imshow(np.sin(np.fft.fftshift(mt.lens_abber_func(k_four, mt.wavelength, 2e-3, 82e-9))), cmap="gray")

    plt.figure(8)
    plt.imshow(np.abs(psi)**2, cmap="gray")
    plt.xlabel("x [Å]")
    plt.ylabel("y [Å]")

    plt.figure(9)
    plt.subplot(1, 2, 1)
    plt.plot(radius_vals, frc1)
    plt.title("FRC for PP")
    plt.xlabel(r"Spatial Frequency [$nm^{-1}$]")
    plt.ylabel("Correlation")

    plt.subplot(1,2,2)
    plt.plot(radius_vals, frc2)
    plt.title("FRC for Scherzer")
    plt.xlabel(r"Spatial Frequency [$nm^{-1}$]")
    plt.ylabel("Correlation")

    plt.figure(10)
    plt.subplot(1, 2, 1)
    plt.plot(radius_vals[:-1], np.sin(-proj_V*mt.sigma_e + np.fft.fftshift(
        mt.lens_abber_func(k_four, mt.wavelength, 2e-3, 0)
    ))[100:199, 100])
    plt.title(r"CTF of PP+Objective Lens with $D=0nm$")
    plt.xlabel(r"Spatial Frequency [$nm^{-1}$]")
    plt.ylabel(r"CTF($k$)")

    plt.subplot(1, 2, 2)
    plt.plot(radius_vals[:-1], np.sin(np.fft.fftshift(
        mt.lens_abber_func(k_four, mt.wavelength, 2e-3, 82e-9)
    ))[100:199, 100])
    plt.title(r"CTF of Objective Lens with Scherzer Defocus")
    plt.xlabel(r"Spatial Frequency [$nm^{-1}$]")
    plt.ylabel(r"CTF($k$)")

    plt.show()




def exitwave_pos(num_points, psi):
    """
    Calulates the probability distribution from the exit wave function
    :param num_points: One sample point
    :param psi: Exit wave function
    :return: One sampled point from the probability distribution of the exit wave
    """

    #psi_magnitude = np.abs(np.fft.fft2(psi))**2
    psi_magnitude = np.abs(np.fft.fftshift(np.fft.fft2(psi)))**2
    #psi_fft = np.clip(psi_magnitude, 0, np.percentile(psi_magnitude, 99))

    psi_fft = psi_magnitude

    #print(np.sum(psi_fft))
    psi_normalized = psi_fft / np.sum(psi_fft)
    #print(np.sum(psi_normalized))

    flattened_psi = psi_normalized.flatten()

    sampled_indices = np.random.choice(flattened_psi.size, size=num_points, p=flattened_psi)

    #sampled_positions = np.unravel_index(sampled_indices, psi_normalized.shape)
    sampled_positions = np.unravel_index(sampled_indices, psi_normalized.shape)


    x_positions, y_positions = sampled_positions

    dkx, min_physical, max_physical = mt.freq_analysis()

    #min_physical = -50e-6
    #max_physical = 50e-6

    grid_size = len(psi[0])

    scale = (max_physical - min_physical) / (grid_size - 1)

    x_positions_rescaled = (x_positions * scale) + min_physical
    y_positions_rescaled = (y_positions * scale) + min_physical


    #plt.figure()
    #plt.scatter(x_positions_rescaled*10**6, y_positions_rescaled*10**6, color='blue', alpha=0.1)
    #plt.xlabel(r"x [$\mu m$]")
    #plt.ylabel(r"y [$\mu m$]")
    #plt.grid(True)
    #plt.show()


    return x_positions_rescaled[0], y_positions_rescaled[0]




def find_Potential_CTF():
    x_vals, y_vals = mt.generate_grid(mt.pots)
    psi = mt.multislice(x_vals, y_vals, 256)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x_vals), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y_vals), d=(voxelsize * angstrom)))
    k = np.sqrt(kx ** 2 + ky ** 2)

    H_0 = mt.objective_transfer_function(k, mt.wavelength, 2e-3, 0, 1)

    psi_with_noise = mt.generate_noise(psi)

    psi_magnitude = np.fft.fftshift(np.fft.fft2(psi_with_noise)*H_0)

    pp_pot1, z_pos_arr = read_Potential_Map("PP_Pot_map_0.txt", "z_pos_0.txt")
    pp_pot2, z_pos_arr2 = read_Potential_Map("PP_Pot_map_1.txt", "z_pos_1.txt")

    tot_pot = np.vstack((pp_pot1, np.rot90(pp_pot2, axes=(1, 2))))
    z_pos = np.array([z_pos_arr, z_pos_arr2])

    proj_pot = np.sum(tot_pot * np.reshape(z_pos, (2 * len(z_pos_arr), 1, 1)), axis=0)

    proj_pot -= np.min(proj_pot)

    psi_magnitude *= np.exp(-1j*mt.sigma_e*proj_pot)

    image = np.abs(np.fft.ifft2(np.fft.ifftshift(psi_magnitude)))**2

    psi_fft = np.clip(np.abs(np.fft.fft2(image)), 0, np.percentile(np.abs(np.fft.fft2(image)), 99))

    plt.figure(1)
    plt.imshow(image, cmap="gray")

    plt.figure(2)
    plt.imshow(np.fft.fftshift(psi_fft), cmap="gray")
    plt.show()

def create_Potential_Maps():
    """
    Creates and saves potential maps to .txt files
    :return: Does not return anything
    """

    electron_array_pp = [
        Electron(charge=-e, keV=20, position=np.array([np.random.normal((i * 0.5e-6 + 0.25e-6) - x_range, 0.25e-6),
                                                       1e-6 * np.random.normal(0, 0.5),
                                                       1e-6 * np.random.normal(0, 0.5)]),
                 velocity=np.array([pp_electron_velocity, 0, 0])) for i in range(pp_electrons)]

    x_vals, y_vals = mt.generate_grid(mt.pots)

    print("Performing Multislice ... ")
    start_mt_time = time.time()
    psi = mt.multislice(x_vals, y_vals, 256)
    end_mt_time = time.time()
    print(f"Multislice Complete! (Time: {end_mt_time - start_mt_time}s)")

    start_pos_x, start_pos_y = exitwave_pos(1, psi)

    beam_electron = [Electron(charge=-e, position=np.array([start_pos_x, start_pos_y, 1e-3]),
                              velocity=np.array([0, 0, -beam_electron_velocity]))]

    all_electrons = electron_array_pp + beam_electron

    for i in range(beam_electrons):

        print(f"Starting simulation for beam electron {i}")

        dt = 7.8124e-14

        new_start_x, new_start_y = exitwave_pos(1, psi)

        potential_calc_size, start_range, end_range = mt.freq_analysis()

        first_time_step_change = False
        second_time_step_change = False
        third_time_step_change = False
        V_arr = []
        z_arr = [beam_electron[0].position[2]]
        start_prop_time = time.time()
        print("Starting Beam propagation ... ")
        while beam_electron[0].position[2] > -1e-3:
            if (30e-6 > beam_electron[0].position[2] > 10e-6) and not first_time_step_change:
                dt /= 4
                first_time_step_change = True
            if (10e-6 > beam_electron[0].position[2] > -1e-6) and not second_time_step_change:
                dt /= 100
                second_time_step_change = True
            if (beam_electron[0].position[2] < -1e-6) and not third_time_step_change:
                dt *= 400
                third_time_step_change = True
            for electrons in all_electrons:
                electrons.rk4_integrator(dt, all_electrons)
            V_arr.append(calculate_potential_slice(elec_array=electron_array_pp, step_size=potential_calc_size,
                                                   start_range=start_range,
                                                   stop_range=end_range, z_val=beam_electron[0].position[2]))
            z_arr.append(beam_electron[0].position[2])

            print("Beam Electron Pos: ", beam_electron[0].position)
        end_prop_time = time.time()
        print(f"Propagation Complete! (Time: {end_prop_time - start_prop_time}s)")

        print("Saving Potential Map and position to files")
        V_arr_3D = np.stack(V_arr)
        PP_pot_path = f"PP_Pot_map_{i}.txt"
        z_pos_path = f"z_pos_{i}.txt"

        if os.path.exists(PP_pot_path):
            with open(PP_pot_path, 'w') as file:

                file.write(f'# Array shape {V_arr_3D.shape}\n')

                for V_slice in V_arr_3D:
                    np.savetxt(file, V_slice, fmt='%e')

                    file.write('# New Slice\n')
        else:
            with open(PP_pot_path, 'x') as file:

                file.write(f'# Array shape {V_arr_3D.shape}\n')

                for V_slice in V_arr_3D:
                    np.savetxt(file, V_slice, fmt='%e')

                    file.write('# New Slice\n')

        if os.path.exists(z_pos_path):
            with open(z_pos_path, 'w') as file:
                np.savetxt(file, np.array(z_arr), fmt='%e')
        else:
            with open(z_pos_path, 'x') as file:
                np.savetxt(file, np.array(z_arr), fmt='%e')

        beam_electron[0].position = np.array([new_start_x, new_start_y, 1e-3])
        beam_electron[0].velocity = np.array([0, 0, -beam_electron_velocity])

def read_Potential_Map(pot_map_file, z_pos_file):
    """
    Used to read potential maps in .txt files
    :param pot_map_file: Path to potential map file
    :param z_pos_file: Path to z pos file
    :return: Arrays containing the potential map and the difference in consecutive z-positions as a (len(z), 1, 1) array
    """
    pot_arr_flat = np.loadtxt(pot_map_file)
    z_pos_arr = np.loadtxt(z_pos_file)

    pot_arr = np.reshape(pot_arr_flat, (len(z_pos_arr)-1, 256, 256))
    z_pos_arr = np.array([np.abs(z_pos_arr[i + 1] - z_pos_arr[i]) for i in range(len(z_pos_arr) - 1)])
    return pot_arr, z_pos_arr

def test_Read_Potential():

    pp_pot1, z_pos_arr = read_Potential_Map("PP_Pot_map_0.txt", "z_pos_0.txt")
    pp_pot2, z_pos_arr2 = read_Potential_Map("PP_Pot_map_1.txt", "z_pos_1.txt")

    #proj_pot_slice1 = pp_pot1 * np.reshape(z_pos_arr, (len(z_pos_arr), 1, 1))
    #proj_pot_slice2 = pp_pot2 * np.reshape(z_pos_arr2, (len(z_pos_arr), 1, 1))

    #proj_pot_slice2 = np.rot90(proj_pot_slice2, axes=(1, 2))

    tot_pot = np.vstack((pp_pot1, np.rot90(pp_pot2, axes=(1, 2))))
    z_pos = np.array([z_pos_arr, z_pos_arr2])

    proj_pot = np.sum(tot_pot * np.reshape(z_pos, (2*len(z_pos_arr), 1, 1)), axis=0)

    proj_pot -= np.min(proj_pot)

    plt.figure(1)
    plt.imshow(proj_pot)
    x_vals, y_vals = mt.generate_grid(mt.pots)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x_vals), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y_vals), d=(voxelsize * angstrom)))
    k_four = np.sqrt(kx ** 2 + ky ** 2)

    plt.figure(2)
    plt.imshow(np.sin(-proj_pot*mt.sigma_e + np.fft.fftshift(mt.lens_abber_func(k_four, mt.wavelength, 2e-3, 0))), cmap="gray")
    plt.show()
def beam_electron_implementation():
    electron_array_pp = [
        Electron(charge=-e, keV=20, position=np.array([np.random.normal((i * 0.5e-6 + 0.25e-6) - x_range, 0.25e-6),
                                                       1e-6 * np.random.normal(0, 0.5),
                                                       1e-6 * np.random.normal(0, 0.5)]),
                 velocity=np.array([pp_electron_velocity, 0, 0])) for i in range(pp_electrons)]

    x_vals, y_vals = mt.generate_grid(mt.pots)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x_vals), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y_vals), d=(voxelsize * angstrom)))
    k_four = np.sqrt(kx ** 2 + ky ** 2)
    print("Performing Multislice ... ")
    start_mt_time = time.time()
    psi = mt.multislice(x_vals, y_vals, 200)
    end_mt_time = time.time()
    print(f"Multislice Complete! (Time: {end_mt_time-start_mt_time}s)")

    start_pos_x, start_pos_y = exitwave_pos(1, psi)

    beam_electron = [Electron(charge=-e, position=np.array([start_pos_x, start_pos_y, 1e-3]),
                             velocity=np.array([0, 0, -beam_electron_velocity]))]

    H_0 = mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, 0, 1)

    psi_pp_interact = np.fft.fftshift(np.fft.fft2(psi)*H_0)

    all_electrons = electron_array_pp + beam_electron

    image_array_slicePot = []
    image_array_endPot = []

    ideal_image = mt.ideal_image()

    for i in range(beam_electrons):

        print(f"Starting simulation for beam electron {i}")

        dt = 7.8124e-14

        new_start_x, new_start_y = exitwave_pos(1, psi)

        potential_calc_size, start_range, end_range = mt.freq_analysis()

        first_time_step_change = False
        second_time_step_change = False
        third_time_step_change = False
        V_arr = []
        start_prop_time = time.time()
        print("Starting Beam propagation ... ")
        while beam_electron[0].position[2] > -1e-3:
            dz_old = beam_electron[0].position[2]
            if (30e-6 > beam_electron[0].position[2] > 10e-6) and not first_time_step_change:
                dt /= 4
                first_time_step_change = True
            if (10e-6 > beam_electron[0].position[2] > -1e-6) and not second_time_step_change:
                dt /= 100
                second_time_step_change = True
            if (beam_electron[0].position[2] < -1e-6) and not third_time_step_change:
                dt *= 400
                third_time_step_change = True
            for electrons in all_electrons:
                electrons.rk4_integrator(dt, all_electrons)
            V_arr.append(calculate_potential_slice(elec_array=electron_array_pp, step_size=potential_calc_size, start_range=start_range,
                                                       stop_range=end_range, z_val=beam_electron[0].position[2])*np.abs(beam_electron[0].position[2]-dz_old))

            print("Beam Electron Pos: ", beam_electron[0].position)
        end_prop_time = time.time()
        print(f"Propagation Complete! (Time: {end_prop_time-start_prop_time}s)")

        beam_electron[0].position = np.array([new_start_x, new_start_y, 1e-3])
        beam_electron[0].velocity = np.array([0, 0, -beam_electron_velocity])

        V_proj_slice_approx = np.sum(V_arr, axis=0)
        V_proj_slice_approx -= np.min(V_proj_slice_approx)

        print("Generating Potential for phase plate second time ...")
        start_ppV_time = time.time()
        end_V, dz = calculate_potential(elec_array=electron_array_pp, step_size=potential_calc_size,
                                          start_range=start_range, stop_range=end_range)
        proj_V_end = np.sum(end_V, axis=2) * dz
        proj_V_end = proj_V_end - np.min(proj_V_end)
        end_ppV_time = time.time()
        print(f"Phase Plate potential calculated! (Time: {end_ppV_time - start_ppV_time}s)")

        image_slicePot = np.abs(np.fft.ifft2(np.fft.ifftshift(psi_pp_interact*np.exp(-1j*mt.sigma_e*V_proj_slice_approx))))**2

        image_endPot = np.abs(np.fft.ifft2(np.fft.ifftshift(psi_pp_interact*np.exp(-1j*mt.sigma_e*proj_V_end))))**2

        image_array_slicePot.append(image_slicePot)
        image_array_endPot.append(image_endPot)

        plt.figure(1, layout='constrained')

        #plt.subplot(1, 3, 1)
        #plt.imshow(proj_V_start, extent=(-50, 50, -50, 50))
        #plt.colorbar()
        #plt.xlabel(r"x [$\mu m$]")
        #plt.ylabel(r"y [$\mu m$]")
        #plt.title("Initial Projected Potential")

        plt.subplot(1, 2, 1)
        plt.imshow(proj_V_end, extent=(-50, 50, -50, 50))
        plt.colorbar()
        plt.xlabel(r"x [$\mu m$]")
        plt.ylabel(r"y [$\mu m$]")



        plt.subplot(1, 2, 2)
        plt.imshow(V_proj_slice_approx, extent=(-50, 50, -50, 50))
        plt.colorbar()
        plt.xlabel(r"x [$\mu m$]")
        plt.ylabel(r"y [$\mu m$]")

        plt.savefig(f"Proj_Pot_calcs_adaptive_timestep_iteration_{i}.png")

        plt.figure(2, layout='constrained')

        plt.subplot(1, 2, 1)
        plt.imshow(np.sin(-proj_V_end*mt.sigma_e + np.fft.fftshift(mt.lens_abber_func(k_four, mt.wavelength, 2e-3, 0))), cmap="gray")


        plt.subplot(1, 2, 2)
        plt.imshow(np.sin(-V_proj_slice_approx*mt.sigma_e + np.fft.fftshift(mt.lens_abber_func(k_four, mt.wavelength, 2e-3, 0))), cmap="gray")


        plt.savefig(f"CTF_calculations_adaptive_timestep_iteration_{i}.png")

        plt.close("all")

        peak_signal_to_noise_ratio1 = PSNR(ideal_image, image_slicePot)
        peak_signal_to_noise_ratio2 = PSNR(ideal_image, image_endPot)
        peak_signal_to_noise_ratio3 = PSNR(ideal_image, np.abs(np.fft.ifft2(np.fft.fft2(psi)*mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, 82e-9, 1)))**2)
        params = [peak_signal_to_noise_ratio1, peak_signal_to_noise_ratio2, peak_signal_to_noise_ratio3]

        with open("PSNR_second_calc.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(params)

        print(f"iteration {i} complete!")

    final_image_slicePot = np.sum(image_array_slicePot, axis=0)
    final_image_endPot = np.sum(image_array_endPot, axis=0)

    final_image_slicePot = final_image_slicePot/np.sum(final_image_slicePot)
    final_image_endPot = final_image_endPot/np.sum(final_image_endPot)

    frc1, r_vals1 = fourier_ring_correlation(final_image_slicePot, ideal_image)
    frc2, r_vals2 = fourier_ring_correlation(final_image_endPot, ideal_image)
    frc3, r_vals3 = fourier_ring_correlation(np.abs(np.fft.ifft2(np.fft.fft2(psi)*mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, 82e-9, 1)))**2, ideal_image)
    radius_vals = np.linspace(0, 4.95, num=len(frc1))

    plt.figure(1)
    plt.imshow(final_image_slicePot, cmap="gray")
    plt.xlabel("x [Å]")
    plt.ylabel("y [Å]")
    plt.title("Final Image Slice Approximation")

    plt.figure(2)
    plt.imshow(final_image_endPot, cmap="gray")
    plt.xlabel("x [Å]")
    plt.ylabel("y [Å]")
    plt.title("Final Image Full Potential")

    plt.figure(3)
    plt.subplot(1, 3, 1)
    plt.plot(radius_vals, frc1)
    plt.xlabel(r"Spatial Frequency [$nm^{-1}$]")
    plt.ylabel("Correlation")
    plt.title("Slice Approximation Summed Image FRC")

    plt.subplot(1, 3, 2)
    plt.plot(radius_vals, frc2)
    plt.xlabel(r"Spatial Frequency [$nm^{-1}$]")
    plt.ylabel("Correlation")
    plt.title("Full Potential Summed Image FRC")

    plt.subplot(1, 3, 3)
    plt.plot(radius_vals, frc3)
    plt.xlabel(r"Spatial Frequency [$nm^{-1}$]")
    plt.ylabel("Correlation")
    plt.title("Scherzer Defocus Image FRC")

    plt.show()
def MSE(ideal, image1):
    """
    Calculates the mean squared error of an image
    :param ideal: Ideal image
    :param image1: Image
    :return: Mean squared error of the image
    """
    gray_scaled_ideal = mt.normalize_and_rescale(ideal)
    gray_scaled_im = mt.normalize_and_rescale(image1)
    squared_error = 0

    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            squared_error += (gray_scaled_ideal[i, j] - gray_scaled_im[i, j])**2

    mean_squared_error = (1/(image1.shape[0]*image1.shape[1]))*squared_error

    return mean_squared_error

def PSNR(ideal, image1):
    """
    Calculates the peak-signal-to-noise-ratio of an image
    :param ideal: Ideal image
    :param image1: Image
    :return: Peak-signal-to-noise-ratio of the image
    """
    MAX_I = 255
    peak_signal_to_noise_ratio = 10 * np.log10((MAX_I**2)/MSE(ideal, image1))

    return peak_signal_to_noise_ratio

def fourier_ring_correlation(image1, image2):
    """
    Calculates the fourier ring correlation of two images
    :param image1: Image one
    :param image2: Image two
    :return: The fourier ring correlation curve and the corresponding spatial frequency vector
    """
    # Check if both images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same dimensions.")

    # Compute the 2D Fourier Transform of both images
    F1 = np.fft.fftshift(np.fft.fft2(image1))
    F2 = np.fft.fftshift(np.fft.fft2(image2))

    # Compute the conjugate product of the two Fourier Transforms
    conj_product = F1 * np.conj(F2)

    # Initialize variables to store the sums of the Fourier components and the conjugate product
    num = np.zeros(image1.shape[0] // 2)
    den1 = np.zeros_like(num)
    den2 = np.zeros_like(num)

    # Calculate the center of the Fourier space
    center = np.array(image1.shape) // 2
    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            # Calculate the distance of each point from the center
            r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(int)
            if r < image1.shape[0] // 2:
                # Sum the conjugate product and the squares of the magnitudes for the denominator
                num[r] += conj_product[x, y]
                den1[r] += np.abs(F1[x, y]) ** 2
                den2[r] += np.abs(F2[x, y]) ** 2


    # Calculate the Fourier Ring Correlation for each ring
    frc = num / np.sqrt(den1 * den2)
    r_vec = np.arange(image1.shape[0] // 2)

    return frc, r_vec

def multislice_phaseplate(psi, pp_pots, dz_vec, spatial_freq):
    """
    Performs multislice through the phase plate
    :param psi: Exit wave function
    :param pp_pots: Phase plate potential
    :param dz_vec: Delta z positions
    :param spatial_freq: Spatial frequencies
    :return: The non shifted fourier transformed exit wave after it has passed through the phase plate
    """

    psi_ft = np.fft.fftshift(np.fft.fft2(psi))

    for i in range(len(dz_vec)):
        propagator = mt.fresnel_propagator(spatial_freq, dz_vec[i])
        proj_pot = (pp_pots[i, :, :]*dz_vec[i]) - np.min(pp_pots[i, :, :]*dz_vec[i]) #Slice wise normalization
        transmission_function = np.exp(-1j * proj_pot * mt.sigma_e)

        psi_ft = np.fft.ifft2(propagator * np.fft.fft2(psi_ft * transmission_function))

    return np.fft.ifftshift(psi_ft)


def multislice_phaseplate_tester():
    x, y = mt.generate_grid(mt.pots)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y), d=(voxelsize * angstrom)))
    k_four = np.sqrt(kx ** 2 + ky ** 2)

    psi = mt.multislice(x, y, 256)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y), d=(voxelsize * angstrom)))
    k_four = np.sqrt(kx ** 2 + ky ** 2)

    H_0 = mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, 0, 1)*CTF_envelope_function(sigma=128)

    pp_beam1, z_pos_1 = read_Potential_Map("PP_Pot_map_0.txt", "z_pos_0.txt")
    pp_beam2, z_pos_2 = read_Potential_Map("PP_Pot_map_1.txt", "z_pos_1.txt")

    pp_beam2 = np.rot90(pp_beam2, axes=(1, 2))

    dz_vec = np.concatenate((z_pos_1, z_pos_2))

    pp_pots = np.vstack((pp_beam1, pp_beam2))

    pp_pots -= np.min(pp_pots)

    psi_with_noise = mt.generate_noise(psi)

    dk, start_k, end_k = mt.freq_analysis()

    rx, ry = np.meshgrid(np.fft.fftfreq(len(x), d=dk),
                         np.fft.fftfreq(len(y), d=dk))

    r = np.sqrt(rx**2 + ry**2)

    psi_after_pp = multislice_phaseplate(psi_with_noise, pp_pots, dz_vec, r)

    image = np.abs(np.fft.ifft2(psi_after_pp * H_0))**2

    image = mt.normalize_and_rescale(image)

    image_fft = np.clip(np.abs(np.fft.fft2(image)), 0, np.percentile(np.abs(np.fft.fft2(image)), 99))

    phase = np.angle(psi_with_noise)

    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.title(r"Image with $Cs = 2mm$ and $D = 0nm$")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title(r"CTF with $Cs = 2mm$ and $D = 0nm$")
    plt.imshow(np.fft.fftshift(image_fft), cmap="gray")

    plt.subplot(1,3,3)
    plt.title(r"Phase of the Exit wave")
    plt.imshow(phase, cmap="gray")

    plt.show()

def multiple_projection_acquisition(filename, base_save_name, num_projections=5, D=None, noise_level=0.2):
    """
    Aquires different projections of an image with defocus and noise. Its important that the length of D and num_projections
    are the same length
    :param filename: The filename with .mrc extension to perform acquisition on
    :param base_save_name: The save filename with .mrc extension
    :param num_projections: Number of projections acquired
    :param D: Defocus value vector
    :param noise_level: Level of added noise
    :return: Projections of an image with defous and noise
    """
    mt.filename = filename

    if D is not None and (len(D) != num_projections):
        raise ValueError("The number of elements in the list D and num_projections must be the same size")


    mt.regenerate_Pots()

    x, y = mt.generate_grid(mt.pots)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y), d=(voxelsize * angstrom)))
    k_four = np.sqrt(kx ** 2 + ky ** 2)

    dk, start_k, end_k = mt.freq_analysis()

    rx, ry = np.meshgrid(np.fft.fftfreq(len(x), d=dk),
                         np.fft.fftfreq(len(y), d=dk))

    r = np.sqrt(rx ** 2 + ry ** 2)

    print("Performing Multislice ... ")
    start_mt_time = time.time()
    psi = mt.multislice(x, y, 256)
    end_mt_time = time.time()
    print(f"Multislice Complete! (Time: {end_mt_time - start_mt_time}s)")

    if D is None:
        D = [50e-9 + i * 50e-9 for i in range(num_projections)]

    for i in range(num_projections):

        psi_with_noise = mt.generate_noise(psi, rel_noise_level=noise_level) #Change this value for more/less noise

        print(f"Starting Image Acquisition iteration {i}")
        start_acq_time = time.time()
        pot_num1 = random.randint(0, 21)
        pot_num2 = random.randint(0, 21)

        pp_beam1, z_pos_1 = read_Potential_Map(f"PP_Pot_map_{pot_num1}.txt", f"z_pos_{pot_num1}.txt")
        pp_beam2, z_pos_2 = read_Potential_Map(f"PP_Pot_map_{pot_num2}.txt", f"z_pos_{pot_num2}.txt")

        pp_beam2 = np.rot90(pp_beam2, axes=(1, 2))

        dz_vec = np.concatenate((z_pos_1, z_pos_2))

        pp_pots = np.vstack((pp_beam1, pp_beam2))

        #pp_pots -= np.min(pp_pots)

        psi_after_pp = multislice_phaseplate(psi_with_noise, pp_pots, dz_vec, r)

        #D = -10e-9 + (5e-9*i)

        H_0 = mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, D[i], 1)*CTF_envelope_function()

        image = np.abs(np.fft.ifft2(psi_after_pp * H_0))**2

        image = mt.normalize_and_rescale(image)

        if os.path.exists(f"{base_save_name}_D_{D[i]:.1e}.mrc"):
            with mrcfile.open(f"{base_save_name}_D_{D[i]:.1e}.mrc", "r+") as mrc:
                mrc.set_data(image)
        else:
            with mrcfile.new(f"{base_save_name}_D_{D[i]:.1e}.mrc") as mrc:
                mrc.set_data(image)

        end_acq_time = time.time()
        print(f"Image Acquisition done! (Time {end_acq_time-start_acq_time}s)")

def multiple_projection_acquisition_with_crossing_beams(filename, base_save_name, num_projections=5, D=None):
    """
    Tester function for projection acquisition with crossing beams instead of stacked
    :param filename:
    :param base_save_name:
    :param num_projections:
    :param D:
    :return:
    """
    mt.filename = filename

    if D is not None and (len(D) != num_projections):
        raise ValueError("The number of elements in the list D and num_projections must be the same size")

    mt.regenerate_Pots()

    x, y = mt.generate_grid(mt.pots)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y), d=(voxelsize * angstrom)))
    k_four = np.sqrt(kx ** 2 + ky ** 2)

    dk, start_k, end_k = mt.freq_analysis()

    rx, ry = np.meshgrid(np.fft.fftfreq(len(x), d=dk),
                         np.fft.fftfreq(len(y), d=dk))

    r = np.sqrt(rx ** 2 + ry ** 2)

    print("Performing Multislice ... ")
    start_mt_time = time.time()
    psi = mt.multislice(x, y, 256)
    end_mt_time = time.time()
    print(f"Multislice Complete! (Time: {end_mt_time - start_mt_time}s)")

    if D is None:
        D = [50e-9 + i * 50e-9 for i in range(num_projections)]

    for i in range(num_projections):

        psi_with_noise = mt.generate_noise(psi, rel_noise_level=0.2) #Change this value for more/less noise

        print(f"Starting Image Acquisition iteration {i}")
        start_acq_time = time.time()
        pot_num1 = random.randint(0, 21)
        pot_num2 = random.randint(0, 21)

        pp_beam1, z_pos_1 = read_Potential_Map(f"PP_Pot_map_{pot_num1}.txt", f"z_pos_{pot_num1}.txt")
        pp_beam2, z_pos_2 = read_Potential_Map(f"PP_Pot_map_{pot_num2}.txt", f"z_pos_{pot_num2}.txt")

        pp_beam2 = np.rot90(pp_beam2, axes=(1, 2))

        dz_vec = z_pos_1

        pp_pots = pp_beam1 + pp_beam2

        #pp_pots -= np.min(pp_pots)

        psi_after_pp = multislice_phaseplate(psi_with_noise, pp_pots, dz_vec, r)

        """if D is None:
            D = 50e-9 + i*50e-9"""

        H_0 = mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, D[i], 1)*CTF_envelope_function()

        image = np.abs(np.fft.ifft2(psi_after_pp * H_0))**2

        image = mt.normalize_and_rescale(image)

        if os.path.exists(f"{base_save_name}_D_{D[i]:.0e}.mrc"):
            with mrcfile.open(f"{base_save_name}_D_{D[i]:.0e}.mrc", "r+") as mrc:
                mrc.set_data(image)
        else:
            with mrcfile.new(f"{base_save_name}_D_{D[i]:.0e}.mrc") as mrc:
                mrc.set_data(image)

        end_acq_time = time.time()
        print(f"Image Acquisition done! (Time {end_acq_time-start_acq_time}s)")

def CTF_envelope_function(size=256, sigma=128):
    """
    Creates the envelope function to be applied to the CTF
    :param size: Size of CTF in pixels
    :param sigma: Standard deviation in pixels
    :return: A Gaussian filter that mimics an envelope function
    """

    filter = np.zeros((size, size))

    center = size // 2
    for i in range(size):
        for j in range(size):
            filter[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))

    return filter

def CTF_envelope_function_tester(size=256, sigma=128):

    filter = CTF_envelope_function(size, sigma)

    x_vals, y_vals = mt.generate_grid(mt.pots)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x_vals), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y_vals), d=(voxelsize * angstrom)))
    k = np.sqrt(kx ** 2 + ky ** 2)

    k_vals = np.fft.fftfreq(len(x_vals), d=(voxelsize * angstrom))
    vals = np.sin(np.fft.fftshift(mt.lens_abber_func(k, mt.wavelength, 2e-3, 82e-9)))

    plt.figure(1)
    plt.contourf(x_vals, y_vals, filter, cmap='viridis')
    plt.colorbar()

    plt.figure(2)
    plt.plot(k_vals[1:len(x_vals) // 2], filter[128, 128:255], label = 'Gaussian Filter')
    plt.plot(k_vals[1:len(x_vals) // 2], (vals*filter)[128, 128:255], label = 'Filtered CTF')
    plt.plot(k_vals[1:len(x_vals) // 2], vals[128, 128:255], label = 'CTF')
    plt.xlabel(r"Spatial Frequency [$nm^{-1}$]")
    plt.ylabel(r"CTF($k$)")
    plt.legend(loc = 'upper left')
    plt.show()

def view_CTF(input_mrc_folder, defocus=63e-9, D_for_plot="0"):
    """
    Plots the spectral averaged CTF in 2D, one in horizontal direction and one in vertical direction
    :param input_mrc_folder: Folder of .mrc files that were acquired with the same defocus value
    :param defocus: The defocus used for plotting a weak phase estimation
    :param D_for_plot: Defocus used for the title of the plots
    """
    folder_path = input_mrc_folder
    mrc_files = glob.glob(os.path.join(folder_path, "*.mrc"))
    ctf_array = []
    for mrc_file in mrc_files:
        with mrcfile.open(mrc_file) as mrc:
            image = mrc.data
        image_fft = np.clip(np.abs(np.fft.fft2(image)), 0, np.percentile(np.abs(np.fft.fft2(image)), 99))

        ctf_array.append(np.fft.fftshift(image_fft))


    stacked_ctf = np.array(ctf_array)

    ctf = np.mean(stacked_ctf, axis=0)
    plt.figure(1)
    plt.imshow(ctf, cmap="gray")

    ctf = (ctf - np.min(ctf))/(np.max(ctf) - np.min(ctf))



    frequencies1 = np.fft.fftshift(np.fft.fftfreq(len(ctf[128, :255]), d=angstrom)) * 1e-10


    plt.figure(2)
    #plt.plot(np.linspace(0, len(ctf[128, 128:255]), num=len(ctf[128, 128:255])), ctf[128, 128:255])
    plt.plot(frequencies1[128:255], ctf[128, 128:255], color='b', label="CTF from image")
    plt.plot(frequencies1[128:255], np.abs(np.sin(mt.objective_transfer_function(frequencies1[128:255]/1e-10, mt.wavelength, 2e-3, defocus)))-0.85, color = 'r',
             label = "Estimated weak phase approximation")
    plt.xlabel("Spatial Frequency [1/Å]")
    plt.ylabel("CTF")
    plt.legend()
    plt.title(f"CTF Horizontal D={D_for_plot}nm")

    ctf_diag = np.diag(ctf)

    frequencies2 = np.fft.fftshift(np.fft.fftfreq(len(ctf_diag), d=angstrom*1/np.sqrt(2))) * 1e-10
    plt.figure(3)
    plt.plot(frequencies2[len(frequencies2)//2:], ctf_diag[len(ctf_diag)//2:], label="CTF from image")
    plt.plot(frequencies2[len(frequencies2)//2:], np.abs(
        np.sin(mt.objective_transfer_function(frequencies2[len(frequencies2)//2:] / 1e-10, mt.wavelength, 2e-3, defocus))) - 0.85,
             color='r',
             label="Estimated weak phase approximation")
    plt.xlabel("Spatial Frequency [1/Å]")
    plt.ylabel("CTF")
    plt.legend()
    plt.title(f"CTF Diagonal D={D_for_plot}nm")
    plt.show()

def generate_all_projections(num_rotations=21, filename="6drv", num_projections=5, D=None, noise_level=0.2):
    """
    Generates all the projections of a protein
    :param num_rotations: Number of rotations
    :param filename: The original protein filename without .mrc extension
    :param num_projections: Number of projections for each rotation
    :param D: List of defocus value
    :param noise_level: The added structural noise level to the image
    """
    for i in range(num_rotations):
        print(f"Starting Projection acquistion iteration {i} of {num_rotations}")
        if i == 0:
            multiple_projection_acquisition(f"{filename}.mrc", f"{filename}_projection", num_projections, D, noise_level)
        else:
            multiple_projection_acquisition(f"{filename}_rotated_{i}.mrc", f"{filename}_rotated_{i}_projection", num_projections, D, noise_level)
        print("Projection acquistion done!")


def effect_of_stacking(filename):

    mt.filename = filename

    mt.regenerate_Pots()

    x, y = mt.generate_grid(mt.pots)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y), d=(voxelsize * angstrom)))
    k_four = np.sqrt(kx ** 2 + ky ** 2)

    dk, start_k, end_k = mt.freq_analysis()

    rx, ry = np.meshgrid(np.fft.fftfreq(len(x), d=dk),
                         np.fft.fftfreq(len(y), d=dk))

    r = np.sqrt(rx ** 2 + ry ** 2)

    print("Performing Multislice ... ")
    start_mt_time = time.time()
    psi = mt.multislice(x, y, 256)
    end_mt_time = time.time()
    print(f"Multislice Complete! (Time: {end_mt_time - start_mt_time}s)")

    pot_num1 = random.randint(0, 21)
    pot_num2 = random.randint(0, 21)

    pp_beam1, z_pos_1 = read_Potential_Map(f"PP_Pot_map_{pot_num1}.txt", f"z_pos_{pot_num1}.txt")
    pp_beam2, z_pos_2 = read_Potential_Map(f"PP_Pot_map_{pot_num2}.txt", f"z_pos_{pot_num2}.txt")

    pp_beam2 = np.rot90(pp_beam2, axes=(1, 2))

    dz_vec = z_pos_1

    dz_vec_stack = np.concatenate((z_pos_1, z_pos_2))

    pp_pots = np.vstack((pp_beam1, pp_beam2))

    dz_vec_stack = np.reshape(dz_vec_stack, (len(dz_vec_stack), 1, 1))

    pp_proj = (np.sum(pp_pots * dz_vec_stack, axis=0) - np.min(np.sum(pp_pots * dz_vec_stack, axis=0)))

    #pp_pots -= np.min(pp_pots)


    psi_after_pp1 = multislice_phaseplate(psi, pp_pots, dz_vec_stack, r)

    pp_pots_cross = pp_beam1 + pp_beam2

    #pp_pots_cross -= np.min(pp_pots_cross)

    psi_after_pp2 = multislice_phaseplate(psi, pp_pots_cross, dz_vec, r)


    psi_after_pp3 = np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(psi)) * np.exp(-1j*pp_proj*mt.sigma_e))


    image_stack = np.abs(np.fft.ifft2(psi_after_pp1 * mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, -40e-9)))**2
    image_cross = np.abs(
        np.fft.ifft2(psi_after_pp2 * mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, -40e-9))) ** 2

    image_single = np.abs(np.fft.ifft2(psi_after_pp3 * mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, -40e-9)))**2

    reference = mt.ideal_image()

    frc_stack, r_vec1 = fourier_ring_correlation(image_stack, reference)
    frc_cross, r_vec2 = fourier_ring_correlation(image_cross, reference)
    frc_single, r_vec3 = fourier_ring_correlation(image_single, reference)

    radius_vals = np.linspace(0, 0.5, num=len(r_vec1))

    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(radius_vals, frc_stack)
    plt.title("Stacked")
    plt.xlabel("Spatial Frequency [1/Å]")
    plt.subplot(1,3,2)
    plt.plot(radius_vals, frc_cross)
    plt.title("Added")
    plt.xlabel("Spatial Frequency [1/Å]")
    plt.subplot(1, 3, 3)
    plt.plot(radius_vals, frc_single)
    plt.title("Single Slice")
    plt.xlabel("Spatial Frequency [1/Å]")
    plt.show()

def plot_For_Potential():
    x, y = mt.generate_grid(mt.pots)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y), d=(voxelsize * angstrom)))
    k_four = np.sqrt(kx ** 2 + ky ** 2)

    pot_num1 = random.randint(0, 21)
    pot_num2 = random.randint(0, 21)

    pp_beam1, z_pos_1 = read_Potential_Map(f"PP_Pot_map_{pot_num1}.txt", f"z_pos_{pot_num1}.txt")
    pp_beam2, z_pos_2 = read_Potential_Map(f"PP_Pot_map_{pot_num2}.txt", f"z_pos_{pot_num2}.txt")

    pp_beam2 = np.rot90(pp_beam2, axes=(1, 2))

    dz_vec = z_pos_1

    dz_vec_stack = np.concatenate((z_pos_1, z_pos_2))

    pp_pots = np.vstack((pp_beam1, pp_beam2))

    dz_vec_stack = np.reshape(dz_vec_stack, (len(dz_vec_stack), 1, 1))

    pp_proj = (np.sum(pp_pots * dz_vec_stack, axis=0) - np.min(np.sum(pp_pots * dz_vec_stack, axis=0)))

    plt.figure(1)
    plt.imshow(pp_proj, extent=(-50, 50, -50, 50))
    plt.xlabel(r"x [$\mu m$]")
    plt.ylabel(r"y [$\mu m$]")
    plt.title("Projected Potential single slice")
    plt.colorbar()

    diag_pots = np.diag(pp_proj)
    diag_pots = diag_pots[len(diag_pots)//2:]

    horizontal_pots = pp_proj[128, 128:255]

    coords_horizontal = np.linspace(0, 50, num=len(horizontal_pots))
    coords_diag = np.linspace(0, 50*np.sqrt(2), num=len(diag_pots))

    plt.figure(2)
    plt.plot(coords_diag, diag_pots * 10**6)
    plt.title("Projected Potential diagonal single slice")
    plt.xlabel(r"Distance [$\mu m$]")
    plt.ylabel(r"Projected Potential [$V \mu m$]")

    plt.figure(3)
    plt.plot(coords_horizontal, horizontal_pots * 10**6)
    plt.title("Projected Potential horizontal single slice")
    plt.xlabel(r"Distance [$\mu m$]")
    plt.ylabel(r"Projected Potential [$V \mu m$]")


    ctf = np.sin(np.fft.fftshift(mt.lens_abber_func(k_four, mt.wavelength, 2e-3, -40e-9)) - (mt.sigma_e*pp_proj))

    ctf_diag = np.diag(ctf)

    plt.figure(4)
    plt.plot(coords_diag/100, ctf_diag[len(ctf_diag)//2:])
    plt.xlabel("spatial frequency [1/Å]")
    plt.title("Weak Phase CTF for D=40nm overfocus")


    plt.figure(5)
    plt.imshow(ctf, cmap="gray")

    plt.show()
def plot_molecule(input_mrc_file):
    """
    Plots 2D projection of the specified .mrc file
    :param input_mrc_file: The name of a 2D .mrc file
    """
    with mrcfile.open(input_mrc_file) as mrc:
        image = mrc.data

    plt.imshow(image, cmap="gray")
    plt.show()


#Write code to run here for encapsulation
if __name__ == "__main__":
    #generate_all_projections(num_rotations=91, noise_level=0.03, D=[-80e-9, -70e-9, -60e-9, -50e-9, -40e-9])
    #multiple_projection_acquisition("6drv_rotated_270.mrc", "6drv_rotated_270_projection", D=[-80e-9, -70e-9, -60e-9, -50e-9, -40e-9], noise_level=0.03)
    #view_CTF("CTF_files_-10", 53e-9, "-10")
    effect_of_stacking("6drv.mrc")
    #plot_For_Potential()
