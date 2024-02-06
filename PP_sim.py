import random
import math
import numpy as np
import matplotlib.pyplot as plt

pp_electrons = 3
beam_electrons = 0

time_step = 1e-18

# constants
k = 8.988 * 10**9
pp_electron_velocity = 18.7e6
m_e = 9.1093837e-31
e = 1.602 * 10 ** -19
c = 299792458  # speed of light
x_range = 0.50e-3
y_range = 0.15
z_range = 0.5

my0 = 1.2566370614*10**-6

angstrom = 1e-10
voxelsize = 0.5  # Ångström
grid_size = 600  # 600x600 pixel grid for image



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
        # Implement acceleration based on Lorentz force equation if needed
        pass

    def rk4_integrator(self, time_step, all_electrons):
        dt = time_step

        k1_v = dt*(self.total_force(all_electrons, self.position, self.velocity)/m_e)
        k1_x = np.multiply(dt,self.velocity)

        k2_v = dt*(self.total_force(all_electrons, self.position + (k1_x/2), self.velocity + (k1_v/2))/m_e)
        k2_x = dt*(self.velocity + (k1_v/2))

        k3_v = dt*(self.total_force(all_electrons, self.position + (k2_x/2), self.velocity + (k2_v/2))/m_e)
        k3_x = dt*(self.velocity + (k2_v/2))

        k4_v = dt*(self.total_force(all_electrons, self.position + k3_x, self.velocity + k3_v)/m_e)
        k4_x = dt*(self.velocity + k3_v)

        self.velocity += (1/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        self.position += (1/6)*(k1_x + 2*k2_x + 2*k3_x + k4_x)

    """ Ta bort?? 
    
    def distance(self, other):
        # Calculate the Euclidean distance between self and other electrons
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        dz = self.position[2] - other.position[2]
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def calculate_distances(self, all_electrons):
        # Calculate distances from self to every other electron
        distances = [self.distance(other) for other in all_electrons]
        self.distance_matrix = np.array(distances)
    """
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

                if( first_run==0):

                    cross_product = np.cross(electron.velocity, unit_vector)

                if (first_run == 1):
                    if(self.keV == 20):
                        self.velocity=(0,electron.keV_to_ms(),0) #initial value

                    cross_product = np.cross(self.velocity, unit_vector)


                b_factor = my0 / (4 * np.pi * distance ** 2)
                F_b = b_factor * cross_product
                magnetic_forces[i] = F_b

        self.magnetic_force_matrix = magnetic_forces

        return np.sum(self.magnetic_force_matrix, axis=0)

    def relative_speed(self, beam_electrons): # måste ev fact checkas med teorin

        if(self.keV==200):

            gamma= (1000*self.keV/(m_e*c**2))+1  # Lorentz factor

            self.velocity= c * math.sqrt(1-(1/(1+gamma**2)))


    def total_force(self, all_electrons, x, v):
        return self.colomb_force(all_electrons=all_electrons, x=x) + self.magnetic_force(all_electrons=all_electrons, x=x, v=v)


#Jag har optimerat potential beräkningen så att nu går det bara igenom en loop istället för 3 (Smyan)
#Den gör samma beräkning men med vectorer och meshgrid istället för index beräkning som använder sig
#av C bibliotek inuti numpy
#Calculate Potential
def calculate_potential(elec_array):

    x_ = np.linspace(0, 0.0005, num=100)
    y_ = np.linspace(-20e-6, 20e-6, num=100)
    z_ = np.linspace(-20e-6, 20e-6, num=100)
    x, y, z = np.meshgrid(x_, y_, z_)

    Vp = np.zeros((len(x), len(y), len(z)))
    dz = z_[1]-z_[0]

    for electron in elec_array:

        elec_pos = np.array(electron.position)

        dist = np.sqrt((x - elec_pos[0]) ** 2 + (y - elec_pos[1]) ** 2 + (z - elec_pos[2]) ** 2)

        Vp += k * (electron.charge / dist)

    return Vp, dz



#Write code to run here for encapsulation (SMYAN)
if __name__ == "__main__":

    # Create an array of Electron objects with random initial positions
    electron_array_pp = [Electron(charge=e, keV=20, position=[np.random.normal(i * 0.5e-6 + 0.25e-6, 0.25e-6),
                                                      1e-6 * np.random.normal(0, 0.5),
                                                      1e-6 * np.random.normal(0, 0.5)]) for i in range(pp_electrons)]

    electron_array_beam = [Electron(charge=e, position=[random.uniform(-x_range, x_range), 0, z_range]) for _ in
                           range(beam_electrons)]


    all_electrons = electron_array_pp + electron_array_beam

    first_run = 1
    runs = pp_electrons+beam_electrons
    for iteration in range(runs):
        for electron in all_electrons:
            if first_run == 1:
               # print(f'First run')
                electron.colomb_force(all_electrons, electron.position)
                electron.magnetic_force(all_electrons, electron.position, electron.velocity)

            if first_run == 0:
                #print(f'NOT first run')
                electron.rk4_integrator(time_step, all_electrons)
                electron.colomb_force(all_electrons, electron.position)
                electron.magnetic_force(all_electrons, electron.position, electron.velocity)

        first_run = 0

    # 3D plot of electron positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from each electron's position for pp_electrons
    x_coords_pp = [electron.position[0] for electron in electron_array_pp]
    y_coords_pp = [electron.position[1] for electron in electron_array_pp]
    z_coords_pp = [electron.position[2] for electron in electron_array_pp]

    # Extract x, y, z coordinates from each electron's position for beam_electrons
    x_coords_beam = [electron.position[0] for electron in electron_array_beam]
    y_coords_beam = [electron.position[1] for electron in electron_array_beam]
    z_coords_beam = [electron.position[2] for electron in electron_array_beam]

    # Plotting electrons from pp_electrons
    ax.scatter(x_coords_pp, y_coords_pp, z_coords_pp, c='b', marker='o', label='pp_electrons')

    # Plotting electrons from beam_electrons
    ax.scatter(x_coords_beam, y_coords_beam, z_coords_beam, c='r', marker='s', label='beam_electrons')

    V, dz = calculate_potential(electron_array_pp)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.imshow(np.sum(V, axis=2)*dz)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()
