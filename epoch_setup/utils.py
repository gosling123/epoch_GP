import fileinput
import os
import sys
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import csv

# Set plot settings
plt.rcParams["figure.figsize"] = (20, 3)
plt.rcParams["figure.figsize"] = [15, 15]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

def read_input(dir, param):
    """
    Reads a specified parameter from the input.deck file in the provided directory.

    dir : str : Directory that contains the input.deck file
    param : str : Parameter to read from input file. Options: 'intensity', 'momentum', 'ppc', 'ne_scale_len'

    Returns:
        float : The requested parameter value
    """
    line = []
    with open(dir+'input.deck') as f:
        found = False
        while not found:
            line = f.readline()
            words = line.split()
            if len(words) < 1:
                continue

            if param == 'intensity':
                if words[0] == "intensity_w_cm2":
                    found = True
                    return float(words[2])

            elif param == 'momentum':
                if words[0] == "range1":
                    found = True
                    return float(words[2][1:-1]), float(words[3][:-1])

            elif param == 'ppc':
                if words[0] == "PPC":
                    found = True
                    return float(words[2])

            elif param == 'ne_scale_len':
                if words[0] == "Ln":
                    found = True
                    return float(words[2])

            else:
                print('Please set param to one of the following as a string: intensity, momentum, ppc or ne_scale_len')
                break

def loss_func(fit, data):
    """
    Computes the loss function to compare the fit data with the simulation data.

    fit : array-like : The fitted data
    data : array-like : The simulation data

    Returns:
        float : The loss value
    """
    N = len(data)
    sum_ = 0
    for i in range(N):
        l = fit[i] - data[i]
        sum_ += l*l

    loss = sum_/N
    return loss

def moving_av(Q, span, period=10):
    """
    Computes the moving average of the input array using scipy's uniform_filter1d function.

    Q : array-like : The data array to compute the moving average on
    span : int : Length of the data
    period : int : The period over which to average (default is 10)

    Returns:
        array-like : The moving average of the input array
    """
    return uniform_filter1d(Q, size=span // period)

def replace_line(line_in, line_out, fname):
    """
    Replaces a line in the specified input file with a new line.

    line_in : str : The original line to be replaced
    line_out : str : The replacement line
    fname : str : The file name to modify

    Returns:
        None
    """
    finput = fileinput.input(fname, inplace=1)
    for i, line in enumerate(finput):
        sys.stdout.write(line.replace(line_in, line_out))
    finput.close()

def append_list_as_row(file_name, list_of_elem):
    """
    Appends a list of elements as a new row in a CSV file.

    file_name : str : The CSV file to append the data to
    list_of_elem : list : List of elements to append as a row

    Returns:
        None
    """
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from the csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def input_deck(I, L_n, L_x, T_e, n_0, output_path, input_file='base_input.deck'):
    """
    Copies a preset input deck into the specified directory and updates values for 
    laser intensity (I), density scale length (Ln), and particles per cell (ppc).

    I : float : Laser intensity (W/cm^2)
    L_n : float : Density scale length (m)
    L_x : float : Particle size (m)
    T_e : float : Electron temperature (keV)
    n_0 : float : Particle density (1/m^3)
    output_path : str : Directory where the input file will be copied to
    input_file : str : Base input file to copy from (default is 'base_input.deck')

    Returns:
        None
    """
    try:
        os.mkdir(output_path)
        print(f'Created {output_path} directory')
    except:
        print(f'{output_path} directory already exists')

    try:
        os.system(f'cp {input_file} ./{output_path}/input.deck')
    except:
        sys.exit('ERROR: Ensure the input_file name is correct as in the input_decks directory')

    replace_line('intensity_w_cm2 =', f'intensity_w_cm2 = {I}', fname=f'./{output_path}/input.deck')
    replace_line('l_n =', f'l_n = {L_n}', fname=f'./{output_path}/input.deck')
    replace_line('l_x =', f'l_n = {L_x}', fname=f'./{output_path}/input.deck')
    replace_line('temperature_in_kev =', f'temperature_in_kev = {T_e}', fname=f'./{output_path}/input.deck')
    replace_line('n_min =', f'n_min = {n_0}', fname=f'./{output_path}/input.deck')
