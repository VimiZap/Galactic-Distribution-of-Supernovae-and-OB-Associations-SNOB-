import numpy as np
import matplotlib.pyplot as plt

import galaxy_tests as gt
import spiral_arm_model as sam



def main():
    # test 1: gt
    gt.run_tests()
    #sam.plot_modelled_emissivity_per_arm([0.18, 0.36, 0.18, 0.28], 'False', False,'cubic', 'false', "output/modelled_emissivity_arms_running_average_7degree45.png")

if __name__ == "__main__":
    main()