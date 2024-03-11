import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
import src.utilities.constants as const
import src.spiral_arm_model as sam
import src.utilities.utilities as ut

    

def test_fractional_contribution(method='linear', readfile='true', h=const.h_spiral_arm, sigma_arm=const.sigma_arm):
    num_rows, num_cols = 2, 2
    # array with fractional contribution of each spiral arm to be testet. Respectively respectively Norma-Cygnus(NC), Perseus(P), Sagittarius-Carina(SA), Scutum-Crux(SC)
    fractional_contribution = [[0.25, 0.25, 0.25, 0.25],
                               [0.18, 0.36, 0.18, 0.28],
                               [0.15, 0.39, 0.15, 0.31],
                               [0.17, 0.34, 0.15, 0.34]]
    # Create subplots for each arm
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    # Adjust the horizontal and vertical spacing
    plt.subplots_adjust(hspace=0.4, wspace=0.12)
    # wspace=0.6 became too tight
    for i in range(num_rows):
        for j in range(num_cols):
            print("Calculating with fractional contribution list: ", i * num_cols + j + 1)
            ax = axes[i, j]
            longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution[i * num_cols + j], method, readfile)
            print(longitudes.shape, densities_as_func_of_long.shape, np.sum(densities_as_func_of_long, axis=0).shape)
            print(np.linspace(0, 100, len(longitudes)))
            print(np.sum(densities_as_func_of_long, axis=0))
            ax.plot(np.linspace(0, 360, len(longitudes)), densities_as_func_of_long[0], label=f"NC. f={fractional_contribution[i * num_cols + j][0]}")
            ax.plot(np.linspace(0, 360, len(longitudes)), densities_as_func_of_long[1], label=f"P. $\ $ f={fractional_contribution[i * num_cols + j][1]}")
            ax.plot(np.linspace(0, 360, len(longitudes)), densities_as_func_of_long[2], label=f"SA. f={fractional_contribution[i * num_cols + j][2]}")
            ax.plot(np.linspace(0, 360, len(longitudes)), densities_as_func_of_long[3], label=f"SC. f={fractional_contribution[i * num_cols + j][3]}")
            ax.plot(np.linspace(0, 360, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), label="Total")
            ax.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_A$ = {sigma_arm} kpc', transform=ax.transAxes, fontsize=8, color='black')
            # Redefine the x-axis labels to match the values in longitudes
            x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
            ax.set_xticks(x_ticks) #np.linspace(0, 360, 13), 
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.set_xlabel("Galactic longitude l (degrees)")
            ax.set_ylabel("Modelled emissivity")
            ax.set_title("Modelled emissivity of the Galactic disk")
            ax.legend()         
    print("Done with plotting. Saving figure...") 
    plt.suptitle('Testing different values for the fractional contribution of each spiral arm')
    plt.savefig("output/test_fractional_contribution2", dpi=1200, bbox_inches='tight')
    #plt.show()  # To display the plot


def test_disk_scale_length(method='linear', readfile='true', fractional_contribution=const.fractional_contribution, sigma_arm=const.sigma_arm):
    # Function to test different values for the disk scale length to see which gives the best fit compared to the data
    disk_scale_lengths = np.array([1.8, 2.1, 2.4, 2.7, 3.0])
    linestyles = np.array(['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 10))])
    for i in range(len(disk_scale_lengths)):
        longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution, method, readfile, h=disk_scale_lengths[i], sigma_arm=sigma_arm)
        plt.plot(np.linspace(0, 100, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), linestyile=linestyles[i], color='black', label=f"$H_\rho$ = {disk_scale_lengths[i]} kpc")
        # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 100, 13), x_ticks)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Modelled emissivity")
    plt.title("Modelled emissivity of the Galactic disk")
    plt.gca().set_aspect('equal')
    # Add parameter values as text labels
    plt.text(0.02, 0.95, fr'$\sigma_A$ = {sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.legend()
    plt.savefig('output/test_disk_scale_length', dpi=1200)
    plt.show()

def test_transverse_scale_length(method='linear', readfile='true', fractional_contribution=const.fractional_contribution, h=const.h_spiral_arm):
    transverse_scale_lengths = np.array([0.25, 0.4, 0.5, 0.6, 0.75])
    linestyles = np.array(['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 10))])
    for i in range(len(transverse_scale_lengths)):
        longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution, method, readfile, h=h, sigma_arm=transverse_scale_lengths[i])
        plt.plot(np.linspace(0, 100, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), linestyile=linestyles[i], color='black', label=f"$\sigma_A$ = {transverse_scale_lengths[i]} kpc")
        # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 100, 13), x_ticks)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Modelled emissivity")
    plt.title("Modelled emissivity of the Galactic disk")
    plt.gca().set_aspect('equal')
    # Add parameter values as text labels
    plt.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.legend()
    plt.savefig('output/transverse_scale_length', dpi=1200)
    plt.show()


def find_max_value_and_index(arr):
    if not arr.any():
        return None, None  # Return None if the array is empty

    max_value = max(arr)
    max_index = np.argmax(arr)

    return max_value, max_index


def find_arm_tangents(fractional_contribution=const.fractional_contribution, gum_cygnus='False', method='cubic', readfile = "false", filename = "output/test_arm_angles/test_arm_start_angle.txt", h=const.h_spiral_arm, sigma_arm=const.sigma_arm):
    # starting angles for the spiral arms, respectively Norma-Cygnus, Perseus, Sagittarius-Carina, Scutum-Crux
    # arm_angles = np.radians([70, 160, 250, 340]) #original
    nc_angle = np.arange(60, 81, 1)
    p_angle = np.arange(150, 171, 1)
    sa_angle = np.arange(240, 261, 1)
    sc_angle = np.arange(330, 351, 1)
    for i in range(len(nc_angle)):
        angles = np.radians(np.array([nc_angle[i], p_angle[i], sa_angle[i], sc_angle[i]]))
        longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution, gum_cygnus, method, readfile, h, sigma_arm, angles)        
        _, max_index_nc = find_max_value_and_index(densities_as_func_of_long[0])
        _, max_index_p = find_max_value_and_index(densities_as_func_of_long[1])
        _, max_index_sa = find_max_value_and_index(densities_as_func_of_long[2])
        _, max_index_sc = find_max_value_and_index(densities_as_func_of_long[3])
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[0], label=f"NC. f={fractional_contribution[0]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[1], label=f"P. $\ $ f={fractional_contribution[1]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[2], label=f"SA. f={fractional_contribution[2]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[3], label=f"SC. f={fractional_contribution[3]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), label="Total")
        print(np.sum(densities_as_func_of_long))
        # Redefine the x-axis labels to match the values in longitudes
        x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
        plt.xticks(np.linspace(0, 100, 13), x_ticks)
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("Galactic longitude l (degrees)")
        plt.ylabel("Modelled emissivity")
        plt.title("Modelled emissivity of the Galactic disk")
        # Add parameter values as text labels
        plt.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_A$ = {sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.text(0.02, 0.9, fr'NII Luminosity = {total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.text(0.02, 0.85, fr'Arm angles: nc={nc_angle[i]}, p={p_angle[i]}, sa={sa_angle[i]}, sc={sc_angle[i]}', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.legend()
        plt.savefig(f'output/test_arm_angles/set_{i}', dpi=1200)
        plt.close()
        # save to file filename
        with open(filename, 'a') as f:
            f.write(f"{nc_angle[i]} {p_angle[i]} {sa_angle[i]} {sc_angle[i]} {np.degrees(longitudes[max_index_nc])} {np.degrees(longitudes[max_index_p])} {np.degrees(longitudes[max_index_sa])} {np.degrees(longitudes[max_index_sc])}\n")
        
        
def find_pitch_angles(fractional_contribution=const.fractional_contribution, gum_cygnus='False', method='cubic', readfile = "false", filename = "output/test_pitch_angles_4/test_pitch_angles_4.txt", h=const.h_spiral_arm, sigma_arm=const.sigma_arm):
    # starting angles for the spiral arms, respectively Norma-Cygnus, Perseus, Sagittarius-Carina, Scutum-Crux
    # arm_angles = np.radians([70, 160, 250, 340]) #original
    # pitch_angles = np.radians(np.array([13.5, 13.5, 13.5, 15.5])) # original
    # Vallées pitch angles: 12.8
    
    #pitch_angles = np.arange(12, 15.6, 0.1)
    pitch_angles = np.arange(13.5, 15.6, 0.1)
    #Arm_Angles = np.array([65, 160, 245, 335]) #1
    #Arm_Angles = np.array([65, 155, 240, 330]) #2
    #Arm_Angles = np.array([65, 160, 250, 330]) #3
    Arm_Angles = np.array([65, 160, 240, 330]) #4
    for i in range(len(pitch_angles)):
        Pitch_Angles = np.radians([pitch_angles[i], pitch_angles[i], pitch_angles[i], pitch_angles[i]])
        longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution, gum_cygnus, method, readfile, h, sigma_arm, np.radians(Arm_Angles), Pitch_Angles)        
        _, max_index_nc = find_max_value_and_index(densities_as_func_of_long[0])
        _, max_index_p = find_max_value_and_index(densities_as_func_of_long[1])
        _, max_index_sa = find_max_value_and_index(densities_as_func_of_long[2])
        _, max_index_sc = find_max_value_and_index(densities_as_func_of_long[3])
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[0], label=f"NC. f={fractional_contribution[0]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[1], label=f"P. $\ $ f={fractional_contribution[1]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[2], label=f"SA. f={fractional_contribution[2]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[3], label=f"SC. f={fractional_contribution[3]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), label="Total")
        print(np.sum(densities_as_func_of_long))
        # Redefine the x-axis labels to match the values in longitudes
        x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
        plt.xticks(np.linspace(0, 100, 13), x_ticks)
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("Galactic longitude l (degrees)")
        plt.ylabel("Modelled emissivity")
        plt.title("Modelled emissivity of the Galactic disk")
        # Add parameter values as text labels
        plt.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_A$ = {sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.text(0.02, 0.9, fr'NII Luminosity = {total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.text(0.02, 0.85, fr'Arm angles: nc={Arm_Angles[0]}, p={Arm_Angles[1]}, sa={Arm_Angles[2]}, sc={Arm_Angles[3]}', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.text(0.02, 0.8, fr'Pitch angles = {pitch_angles[i]}', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.legend()
        plt.savefig(f'output/test_pitch_angles_4/set_{i}', dpi=1200)
        plt.close()
        # save to file filename
        with open(filename, 'a') as f:
            f.write(f"{pitch_angles[i]} {np.degrees(longitudes[max_index_nc])} {np.degrees(longitudes[max_index_p])} {np.degrees(longitudes[max_index_sa])} {np.degrees(longitudes[max_index_sc])}\n")
