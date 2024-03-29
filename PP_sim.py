import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import mulitslice as mt
import scipy.spatial.distance as distance
from scipy.integrate import simps
import csv


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
grid_size = 200  # 200x200 pixel grid for image

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

    def accelerate(self, electric_field, magnetic_field):

        pass

    def rk4_integrator(self, time_step, all_electrons):
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

        self.velocity += time_step*(self.total_force(all_electrons, self.position, self.velocity)/m_e)
        self.position += time_step*self.velocity

    def colomb_force(self, all_electrons, x):
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
        return math.sqrt(2 * self.keV *1000*e/ m_e)

    def magnetic_force(self, all_electrons, x, v):  # Biot–Savart law
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


    def relative_speed(self, beam_electrons): # måste ev fact checkas med teorin

        if(self.keV==200):

            gamma = (1000*self.keV/(m_e*c**2))+1  # Lorentz factor

            self.velocity = c * math.sqrt(1-(1/(1+gamma**2)))


    def total_force(self, all_electrons, x, v):
        return self.colomb_force(all_electrons=all_electrons, x=x) + self.magnetic_force(all_electrons=all_electrons, x=x, v=v)


#Jag har optimerat potential beräkningen så att nu går det bara igenom en loop istället för 3 (Smyan)
#Den gör samma beräkning men med vectorer och meshgrid istället för index beräkning som använder sig
#av C bibliotek inuti numpy
#Calculate Potential
def calculate_potential(elec_array, step_size, start_range, stop_range):

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

"""
def simulate_beam_electrons():
    global beam_electron_index, frames_passed, beam_positions

    while beam_electron_index < len(electron_array_beam):
        electron = electron_array_beam[beam_electron_index]
        for _ in range(10):  # Perform 10 iterations of RK4 integration
            electron.rk4_integrator(all_electrons=electron_array_pp + [electron], time_step=time_step)
        beam_positions.append(electron.position)
        beam_electron_index += 1
"""

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

    scale = (max_physical - min_physical) / (grid_size - 1)

    x_positions_rescaled = (x_positions * scale) + min_physical
    y_positions_rescaled = (y_positions * scale) + min_physical

    """
    plt.figure()
    plt.scatter(x_positions_rescaled*10**6, y_positions_rescaled*10**6, color='blue', alpha=0.1)
    plt.xlabel(r"x [$\mu m$]")
    plt.ylabel(r"y [$\mu m$]")
    plt.grid(True)
    plt.show()
    """

    return x_positions_rescaled[0], y_positions_rescaled[0]




def find_Potential_CTF():
    electron_array_pp = [
                Electron(charge=-e, keV=20, position=np.array([np.random.normal((i * 0.5e-6 + 0.25e-6) - x_range, 0.25e-6),
                                                       1e-6 * np.random.normal(0, 0.5),
                                                       1e-6 * np.random.normal(0, 0.5)]),
                 velocity=np.array([0, 0, 0])) for i in range(pp_electrons)]

    potential_calc_size, start_range, end_range = mt.freq_analysis()

    V, dz = calculate_potential(electron_array_pp, potential_calc_size, start_range, end_range)

    proj_V = np.sum(V, axis=2)*dz

    x, y = mt.generate_grid(mt.pots)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y), d=(voxelsize * angstrom)))
    k_four = np.sqrt(kx ** 2 + ky ** 2)

    limit_freq = (2 / 3) * (0.5 / (voxelsize*angstrom))

    lp_filter = np.sqrt(kx ** 2 + ky ** 2) <= limit_freq

    mean = 0
    std_dev = 0.01 * 2.5e-5

    random_gaussian_values = np.random.normal(loc=mean, scale=std_dev, size=np.shape(x))

    # Normalize the wavefunction
    normalization_factor = np.sqrt(np.sum(np.abs(random_gaussian_values) ** 2))
    normalized_wavefunction = random_gaussian_values / normalization_factor

    filtered_psi_vals = np.fft.fft2(normalized_wavefunction) * lp_filter

    psi_vals = filtered_psi_vals*mt.objective_transfer_function(k_four, mt.wavelength, 2e-3, 0, 1)



    CTF = np.fft.fftshift(psi_vals)*np.exp(-1j*mt.sigma_e*proj_V)



    plt.imshow(np.abs(CTF)**2, cmap="gray")
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


        dt = 7.8124e-14

        new_start_x, new_start_y = exitwave_pos(1, psi)

        potential_calc_size, start_range, end_range = mt.freq_analysis()

        """print("Generating Potential for phase plate ...")
        start_ppV_time = time.time()
        start_V, dz = calculate_potential(elec_array=electron_array_pp, step_size=potential_calc_size, start_range=start_range, stop_range=end_range)
        proj_V_start = np.sum(start_V, axis=2)*dz
        proj_V_start = proj_V_start - np.min(proj_V_start)
        end_ppV_time = time.time()
        print(f"Phase Plate potential calculated! (Time: {end_ppV_time - start_ppV_time}s)")"""
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

    gray_scaled_ideal = mt.normalize_and_rescale(ideal)
    gray_scaled_im = mt.normalize_and_rescale(image1)

    squared_error = 0

    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            squared_error += (gray_scaled_ideal[i, j] - gray_scaled_im[i, j])**2

    mean_squared_error = (1/(image1.shape[0]*image1.shape[1]))*squared_error

    return mean_squared_error

def PSNR(ideal, image1):


    MAX_I = 255

    peak_signal_to_noise_ratio = 10 * np.log10((MAX_I**2)/MSE(ideal, image1))

    return peak_signal_to_noise_ratio

def fourier_ring_correlation(image1, image2):
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
    frc = np.abs(num) / np.sqrt(den1 * den2)
    r_vec = np.arange(image1.shape[0] // 2)

    return frc, r_vec
#Write code to run here for encapsulation (SMYAN)
if __name__ == "__main__":
    #tester_1()
    pp_stationary()
    #find_Potential_CTF()
    #beam_electron_implementation()

