import numpy as np
import mrcfile
import matplotlib.pyplot as plt

#Constants
h = 6.62607015*10**-34 #plancks constant in SI units
e = 1.60217663*10**-19 #Electron charge in C
V_a = 200*10**3 #Acceleration voltage in V
c = 299792458 #Speed of light in m/s
m_e = 9.1093937*10**-31 #Electron mass in kg
wavelength = (h*c)/np.sqrt((e*V_a)**2 + 2*e*V_a*m_e*(c**2)) #Relativistically corrected de Broglie wavelength for fast moving electrons

v = np.sqrt(2*(e*V_a)/m_e)
m_relativistic = (1/np.sqrt(1-((v**2)/(c**2))))*m_e

sigma_e = 2*np.pi*m_relativistic*e*wavelength/(h**2) #Interaction parameter

#Sets the file for the simulation
#When performing multiple simulations this variable value should be changed dynamically
filename = "4xcd_200.mrc"

#Load the file to run
with mrcfile.open(filename) as mrc:
    pots = mrc.data

angstrom = 1e-10

#Sets the voxelsize or resolution of the simulation (This myst match the voxelsize of the .mrc files)
voxelsize = 1 #Ångström


padding_size = 50  # This is an example value, adjust as needed
padded_pots = np.pad(pots, pad_width=padding_size, mode='constant', constant_values=0)

def generate_grid(V):
    """
    Generates the grid for the multislice simulation
    :param V: The potential map of the molecule to be imaged
    :return: The meshgrid X, Y for the simulation
    """
    grid_size = np.array(V.shape[:2])  # Assuming V is 3D and grid size is based on the first two dimensions
    #x = np.arange(0, grid_size[0]//2, voxelsize) * angstrom
    #y = np.arange(0, grid_size[1]//2, voxelsize) * angstrom

    x = np.arange(0, int(grid_size[0]*voxelsize), voxelsize) * angstrom
    y = np.arange(0, int(grid_size[1]*voxelsize), voxelsize) * angstrom
    X, Y = np.meshgrid(x, y)
    return X, Y

def calculate_proj_pot(V, nslice):
    """
    Calculates the slice-wise projected potential of the molecule
    :param V: The potential map of the molecule to be imaged
    :param nslice: The number of slices to be used in the multislice simulation
    :return: The projected potential in Vm for each slice in an array and the slice thickness
    """
    dz = len(V)//nslice
    V_z = []

    for i in range(nslice):
        start_z = i*dz
        end_z = start_z + dz
        if i == nslice - 1:
            end_z = len(V)


        integrated_values = (voxelsize*angstrom)*np.sum(V[start_z:end_z, :, :], axis=0)
        V_z.append(integrated_values)

    return V_z, dz

def calculate_transmission_function(proj_pot):
    """
    Calculates the tranmission function used in the multislice simulation for each slice
    :param proj_pot: The projected potential for one slice of the specimen
    :return: The bandwidth limited transmission function for the slice
    """
    t_n = []
    for pot_slice in proj_pot:
        # Calculate transmission function
        t_n_val = np.exp(1j * sigma_e * pot_slice)

        # Apply Fourier transform
        ft_t_n_val = np.fft.fft2(t_n_val)
        ft_t_n_shifted = np.fft.fftshift(ft_t_n_val)  # Shift to center the zero frequency

        # Define the frequency domain grid
        ny, nx = pot_slice.shape
        dx = angstrom * voxelsize
        kx = np.fft.fftfreq(nx, d=dx)
        ky = np.fft.fftfreq(ny, d=dx)
        kx, ky = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky))
        k = np.sqrt(kx ** 2 + ky ** 2)

        # Determine the frequency limit (2/3 of Nyquist frequency)
        limit_freq = (2/3) * (0.5 / dx)

        # Apply bandwidth limiting

        mask = k <= limit_freq
        band_limited_ft = ft_t_n_shifted * mask

        # Shift back and apply inverse Fourier transform
        band_limited_ft_shifted_back = np.fft.ifftshift(band_limited_ft)
        limited_t_n_val = np.fft.ifft2(band_limited_ft_shifted_back)

        t_n.append(limited_t_n_val)


    return t_n


def fresnel_propagator(k, dz):
    """
    Calculates the fresnel propagator for one slice
    :param k: The spatial frequency coordinates for the multislice grid
    :param dz: The slice thickness
    :return: The propagator function for one slice in reciprocal space
    """
    # Apply the Fresnel propagation formula
    p = np.exp(-1j * np.pi * wavelength * (k ** 2) * dz)

    return p



def multislice(x, y, nslices):

    """
    Performs the multislice algorithm to acquire an exit wave
    :param x: The x-coordinates for the grid (as a meshgrid)
    :param y: The y-coordinates for the grid (as a meshgrid)
    :param nslices: The number of slices for the multislice simulation
    :return: The exit wave from the specimen
    """
    psi_0_unnormalized = np.ones_like(x)
    number_of_points = x.size  # Total number of points in the grid
    normalization_factor = 1 / np.sqrt(number_of_points)
    psi = psi_0_unnormalized * normalization_factor

    projpot, dz = calculate_proj_pot(V=pots, nslice=nslices)



    t_n = calculate_transmission_function(projpot)
    slice_size = dz*angstrom*voxelsize

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x), d=(voxelsize*angstrom)), np.fft.fftfreq(len(y), d=(voxelsize*angstrom)))
    k = np.sqrt(kx ** 2 + ky ** 2)


    for i in range(nslices):
        psi = np.fft.ifft2(fresnel_propagator(k, slice_size)*np.fft.fft2(t_n[i] * psi))
    return psi


def lens_abber_func(k, lambda_, Cs, delta_f):
    """
    Calculates the lens abberation function chi(k)
    :param k: Spatial frequencies
    :param lambda_: Electron wavelength
    :param Cs: Spherical abberation coefficient
    :param delta_f: Defocus value
    :return: lens abberation function
    """
    return np.pi * lambda_ * (k ** 2) * (0.5 * Cs * lambda_ ** 2 * k ** 2 - delta_f)


def objective_transfer_function(k, lambda_, Cs, delta_f, A_k=1):
    """
    Calculates the objective lens transfer function for the weak phase approximation
    :param k: Spatial frequencies
    :param lambda_: Electron wavelength
    :param Cs: Spherical abberation coefficient
    :param delta_f: Defocus value
    :param A_k: Apperture coefficient (Use as 1)
    :return: The objective lens transfer function for the weak phase approximation
    """
    chi_k = lens_abber_func(k, lambda_, Cs, delta_f)
    return np.exp(1j * chi_k) * A_k


def normalize_and_rescale(data):
    """
    Normalizes and rescales an image to grayscale
    :param data: Image to be rescaled
    :return: grayscaled normalized image
    """
    # Normalize data to [0, 1]
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # Rescale to [0, 255] for grayscale
    rescaled_data = (normalized_data * 255).astype(np.uint8)
    return rescaled_data
def test_mult():
    x, y = generate_grid(pots)
    psi = multislice(x, y, 200)

    plt.figure(1)
    plt.imshow(np.abs(psi)**2, cmap="gray")



    kx, ky = np.meshgrid(np.fft.fftfreq(len(x), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y), d=(voxelsize * angstrom)))
    k = np.sqrt(kx ** 2 + ky ** 2)

    H_0 = objective_transfer_function(k, wavelength, 2e-3, 82e-9, 1)
    # plt.figure(2)
    # plt.imshow(np.abs(H_0), cmap="gray_r")
    Im = np.fft.fftshift(np.fft.fft2(psi)) * np.fft.fftshift(H_0)



    plt.figure(2)
    plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(Im))) ** 2, cmap="gray")
    plt.xlabel("x [Å]")
    plt.ylabel("y [Å]")

    plt.figure(3)
    plt.imshow(np.sin(np.fft.fftshift(lens_abber_func(k, wavelength, 2e-3, 82e-9))), cmap="gray")

    plt.figure(4)
    plt.imshow(np.clip(np.abs(np.fft.fftshift(np.fft.fft2(psi)))**2, 0,np.percentile(np.abs(np.fft.fftshift(np.fft.fft2(psi)))**2,99)),cmap="gray")

    plt.figure(5)
    plt.imshow(np.log(1+1000*np.abs(np.fft.fftshift(np.fft.fft2(psi)))**2), cmap="gray")

    plt.show()

def generate_noise(wavefunction):
    """
    Generates noise in an arbitrary wavefunction. Is to be used to simulate noise from the specimen.
    :param wavefunction: An arbitrary 2D wavefunction
    :return: wavefunction with complex-valued gaussian noise
    """
    noise_mean = 0
    noise_std_imag = 0.1 * np.mean(np.imag(wavefunction))
    noise_std_real = 0.1 * np.mean(np.real(wavefunction))

    noise_real = np.random.normal(loc=noise_mean, scale=noise_std_real, size=wavefunction.shape)
    noise_imag = np.random.normal(loc=noise_mean, scale=noise_std_imag, size=wavefunction.shape)

    complex_noise = noise_real + 1j * noise_imag

    noisy_wavefunction = wavefunction + complex_noise

    return noisy_wavefunction
def test_mult_with_noise_and_rescaling():
    x, y = generate_grid(pots)
    psi = multislice(x, y, 200)

    psi_noisy = generate_noise(psi)

    kx, ky = np.meshgrid(np.fft.fftfreq(len(x), d=(voxelsize * angstrom)),
                         np.fft.fftfreq(len(y), d=(voxelsize * angstrom)))
    k = np.sqrt(kx ** 2 + ky ** 2)

    H_0 = objective_transfer_function(k, wavelength, 2e-3, 82e-9, 1)
    # plt.figure(2)
    # plt.imshow(np.abs(H_0), cmap="gray_r")
    Im = np.fft.fft2(psi_noisy) * H_0

    plt.subplot(1,3,1)
    plt.imshow(np.abs(psi)**2, cmap="gray")

    plt.subplot(1,3,2)
    plt.imshow(np.abs(psi_noisy)**2, cmap="gray")

    plt.subplot(1,3,3)
    plt.imshow(np.abs(np.fft.ifft2(Im))**2, cmap="gray")

    plt.show()

# d = k*lambda*z
def freq_analysis(focal_length=4e-3):
    """
    Creates the map in reciprocal space for the phase plate
    :return: The pixel spacing in reciprocal space, the leftmost and rightmost x-coordinates in reciprocal space.
    """
    x, y = generate_grid(pots)

    #focal_length = 4e-3 #m

    kx = np.fft.fftfreq(len(x), d=(voxelsize*angstrom))
    ky = np.fft.fftfreq(len(y), d=(voxelsize*angstrom))

    return (kx[1]-kx[0])*wavelength*focal_length, np.fft.fftshift(kx)[0]*wavelength*focal_length, np.fft.fftshift(kx)[-1]*wavelength*focal_length


def ideal_image():

    V, dz = calculate_proj_pot(V=pots, nslice=200)

    #plt.imshow(np.sum(V, axis=0), cmap="gray")
    #plt.xlabel("x [Å]")
    #plt.ylabel("y [Å]")
    #plt.show()

    return np.sum(V, axis=0)
if __name__ == "__main__":
    #test_mult()
    #print(freq_analysis())
    #ideal_image()
    #print(np.sqrt((4/3)*2e-3*wavelength))
    test_mult_with_noise_and_rescaling()
