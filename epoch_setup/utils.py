from typing import ForwardRef
import fileinput
import os
import sys
from nbformat import read
import numpy as np
import sdf
from scipy import constants
from scipy.optimize import brentq
import glob
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
import csv
import json
import time



plt.rcParams["figure.figsize"] = (20,3)
plt.rcParams["figure.figsize"] = [15, 15]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
# Constants

c = constants.c
eps0 = constants.epsilon_0
me = constants.m_e
e = constants.e
kB = constants.k
keV_to_K = (e*1e3)/kB
mu0 = constants.mu_0
pi = np.pi
pico = 1e-12
micron = 1e-6
nano = 1e-9
J_tot_KeV = 6.242e+15


## read_input
#
# Read inputs in input.deck in chosen directory
# @param dir : Directory which holds input.deck file you want to read (str)
# @param param : Specific paramter to read from input file ('intensity', 'momentum', 'ppc' or 'ne_scale_len')
def read_input(dir, param):
    
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
                    print('Please set param to one of the following as a string \
                           intensity, momentum, ppc or ne_scale_len')
                    break




## loss_func
#
# Calculates loss function to compare fits
# @param fit : fit data
# @param data : sim data
def loss_func(fit, data):
    N = len(data)
    sum_ = 0
    for i in range(N):
        l = fit[i] - data[i]
        sum_ += l*l
    
    loss = sum_/N

    return loss


## moving_av
#
# Finds movng average of an array using scipys uniform_filter1d function
# @param Q : Data array 
# @param span : length of data 
# @param period : period to average over
def moving_av(Q, span, period = 10):
    return uniform_filter1d(Q, size = span // period)



## replace_line
#
# Function rewrite line in input.deck via python
# @param line_in : original line in input.deck 
# @param line_out : repacement of line_in in input.deck
def replace_line(line_in, line_out, fname):
  finput = fileinput.input(fname, inplace=1)
  for i, line in enumerate(finput):
    sys.stdout.write(line.replace(line_in, line_out))
  finput.close()


## append_list_as_row
#
# Append data to csv file (for appending I and I_SRS result)
# @param file_name : file name of csv file 
# @param list_of_elem : list to write to csv file 
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


## input_deck
#
# Copies a preset input deck into chosen directory setting the values of
# laser intensity (I), density scale length (Ln) and particles per cell (ppc).
# @param I  Laser intensity (W/cm^2)  
# @param Ln  Density scale length (m)
# @param ppc  Paricles per cell
# @param dir  Directory name
# @param input_file  Input file base to copy from input_decks folder
def input_deck(I, L_n, L_x, T_e, n_0, output_path, input_file = 'base_input.deck'): 

    try:
        os.mkdir(output_path)
        print(f'Created {output_path} directory')
    except:
        print(f'{output_path} directory already exists')

    try:
        os.system(f'cp {input_file} ./{output_path}/input.deck')
    except:
        sys.exit('ERROR: Ensure the input_file name is correct as in the input_decks directory')


    replace_line('intensity_w_cm2 =', f'intensity_w_cm2 = {I}', fname = f'./{output_path}/input.deck')
    replace_line('l_n =', f'l_n = {L_n}', fname = f'./{output_path}/input.deck')
    replace_line('l_x =', f'l_n = {L_x}', fname = f'./{output_path}/input.deck')
    replace_line('temperature_in_kev =', f'temperature_in_kev = {T_e}', fname = f'./{output_path}/input.deck')
    replace_line('n_min =', f'n_min = {n_0}', fname = f'./{output_path}/input.deck')
