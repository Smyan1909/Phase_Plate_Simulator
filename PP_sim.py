import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import mulitslice as mt
import scipy.spatial.distance as distance


pp_electrons = 200
beam_electrons = 0

time_step = 1e-13

# constants
k = 8.988 * 10**9
pp_electron_velocity = 18.7e6
beam_electron_velocity = 18.7e7
m_e = 9.1093837e-31
e = 1.602 * 10 ** -19
c = 299792458  # speed of light
x_range = 0.05e-3
y_range = 0.05e-3
z_range = 0.05e-3

my0 = 1.2566370614*10**-6

angstrom = 1e-10
voxelsize = 1  # Ångström
grid_size = 400  # 600x600 pixel grid for image



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
    z_ = np.arange(start_range, stop_range+step_size, step_size)
    x, y, z = np.meshgrid(x_, y_, z_, sparse=True)

    Vp = np.zeros((len(x_), len(y_), len(z_)))
    dz = z_[1]-z_[0]

    for electron in elec_array:

        elec_pos = np.array(electron.position)

        dist = np.sqrt((x - elec_pos[0]) ** 2 + (y - elec_pos[1]) ** 2 + (z - elec_pos[2]) ** 2)

        Vp += k * (electron.charge / dist)



    return Vp, dz

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

    """
    print("Positions of Beam Electrons after 10 iterations:")
    for i, pos in enumerate(beam_positions):
        print(f"Beam Electron {i + 1}: {pos}")

    beam_positions_np = np.array(beam_positions)
    has_z_above_zero = np.any(beam_positions_np[:, 2] > 0)
    print(has_z_above_zero)

    """

    """
    for electron in all_electrons:
            print(electron.position)
            electron.rk4_integrator(time_step=time_step, all_electrons=all_electrons)
            print(electron.position)

    first_run   = 0
    """
    # 3D plot of electron positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Extract x, y, z coordinates from each electron's position for pp_electrons
    x_coords_pp = [electron.position[0] for electron in electron_array_pp]
    y_coords_pp = [electron.position[1] for electron in electron_array_pp]
    z_coords_pp = [electron.position[2] for electron in electron_array_pp]

    """
    # Extract x, y, z coordinates from each electron's position for beam_electrons
    x_coords_beam = [electron.position[0] for electron in electron_array_beam]
    y_coords_beam = [electron.position[1] for electron in electron_array_beam]
    z_coords_beam = [electron.position[2] for electron in electron_array_beam]
    """

    # Plotting electrons from pp_electrons
    ax.scatter(x_coords_pp, y_coords_pp, z_coords_pp, c='b', marker='o', label='pp_electrons')

    """
    # Plotting electrons from beam_electrons
    ax.scatter(x_coords_beam, y_coords_beam, z_coords_beam, c='r', marker='s', label='beam_electrons')
    """

    #V, dz = calculate_potential(electron_array_pp,)

    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(111)
    #ax2.imshow(np.sum(V, axis=2) * dz)

    #ani = FuncAnimation(fig, update, frames=range(200), fargs=(all_electrons, time_step, ax))

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
    plt.xlabel(f"x [$\mu m$]")
    plt.ylabel(f"y [$\mu m$]")

    plt.figure(5)
    plt.imshow(np.sin(proj_V*mt.sigma_e), cmap="gray")

    plt.figure(6)
    plt.imshow(np.sin(proj_V*mt.sigma_e + np.fft.fftshift(mt.lens_abber_func(k_four, mt.wavelength, 2e-3, 0))), cmap="gray")

    plt.figure(7)
    plt.imshow(np.sin(np.fft.fftshift(mt.lens_abber_func(k_four, mt.wavelength, 2e-3, 82e-9))), cmap="gray")

    plt.figure(8)
    plt.imshow(np.abs(psi)**2, cmap="gray")
    plt.xlabel("x [Å]")
    plt.ylabel("y [Å]")

    plt.show()

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

    mean = 0
    std_dev = 1

    random_gaussian_values = np.random.normal(loc=mean, scale=std_dev, size=np.shape(x))

    # Normalize the wavefunction
    normalization_factor = np.sqrt(np.sum(np.abs(random_gaussian_values) ** 2))
    normalized_wavefunction = random_gaussian_values / normalization_factor


    CTF = np.fft.fftshift(np.fft.fft2(normalized_wavefunction))*np.exp(-1j*mt.sigma_e*proj_V)

    print(CTF)

    plt.imshow(np.angle(CTF), cmap="gray")
    plt.show()
#Write code to run here for encapsulation (SMYAN)
if __name__ == "__main__":
    tester_1()
    #pp_stationary()
    #find_Potential_CTF()
