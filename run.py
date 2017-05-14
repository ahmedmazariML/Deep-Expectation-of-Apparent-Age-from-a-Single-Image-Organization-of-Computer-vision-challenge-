#!/usr/bin/env python

# Usage: python run.py input_dir output_dir

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This code can be used either to submit RESULTS of a TRAINED MODE 
# depostited in the saved_results/ subdirectory or for CODE submission of
# an untrained model. The template of code is sample_code/classifier.py
#
# The input directory input_dir contains 5 subdirectories named by dataset,
# including:
#    dataname/dataname_feat.name          -- the feature names (column headers of data matrix)
# 	dataname/dataname_feat.type          -- the feature type "Numerical", "Binary", or "Categorical" (Note: if this file is abscent, get the feature type from the dataname.info file)
#    dataname/dataname_label.name          -- the label names (column headers of the solution matrix)
# 	dataname/dataname_public.info        -- parameters of the data and task, including metric and time_budget
# 	dataname/dataname_test.data          -- training, validation and test data (solutions/target values are given for training data only)
# 	dataname/dataname_train.data
# 	dataname/dataname_train.solution
# 	dataname/dataname_valid.data
#
# The output directory will receive the predicted values (no subdirectories):
# 	dataname_test.predict            
# 	dataname_valid.predict
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 
#
# Main contributors: Isabelle Guyon and Arthur Pesah, March-October 2014
# Lukasz Romaszko April 2015
# Originally inspired by code code: Ben Hamner, Kaggle, March 2013
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013
# Last modifications Isabelle Guyon, November 2015

# =========================== BEGIN OPTIONS ==============================
# Verbose mode: 
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
############## 
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there may be several datasets). 
# The code should keep track of time spent and NOT exceed the time limit 
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 500 

# Maximum number of cycles, number of samples, and estimators
#############################################################
# Your training algorithm may be fast, so you may want to limit anyways the 
# number of points on your learning curve (this is on a log scale, so each 
# point uses twice as many time than the previous one.)
# The original code was modified to do only a small "time probing" followed
# by one single cycle. We can now also give a maximum number of estimators 
# (base learners).
max_cycle = 1 
max_estimators = float('Inf')
max_samples = 50000

# Save your model
#################
save_model = True

# ZIP your results and code
###########################
# You can create a code submission archive, ready to submit, with zipme = True.
# This is meant to be used on your LOCAL server.
import datetime
zipme = True # use this flag to enable zipping of your code submission
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
submission_filename = '../sample_submission_' + the_date

# I/O defaults
##############
# If true, the previous output directory is not overwritten, it changes name
save_previous_results = False
overwrite_output = True # save space
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "/Users/isabelleguyon/Documents/Projects/ParisSaclay/Projects/ChaLab/Examples/iris/"
default_input_dir = root_dir + "input_data_1_2/"
default_output_dir = root_dir + "scoring_input_1_2/res"

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Version of the sample code
version = 4 

# General purpose functions
import time
overall_start = time.time()         # <== Mark starting time
import os
from sys import argv, path
import numpy as np
import gc

# Our directories
# Note: On cadalab, there is an extra sub-directory called "program"
# Keave this stuff "as is"
running_on_codalab = False
run_dir = os.path.abspath(".")
codalab_run_dir = os.path.join(run_dir, "program")
if os.path.isdir(codalab_run_dir): 
    run_dir=codalab_run_dir
    running_on_codalab = True
    print "Running on Codalab!"
lib_dir = os.path.join(run_dir, "sample_code")
res_dir = os.path.join(run_dir, "res")

# Our libraries  
path.append (run_dir)
path.append (lib_dir)
import data_io                       # general purpose input/output functions
from data_io import vprint           # print only in verbose mode
from data_manager import DataManager # load/save data and get info about them
from classifier import Classifier    # example models from scikit learn

if debug_mode >= 4 or running_on_codalab: # Show library version and directory structure
    data_io.show_version()
    data_io.show_dir(run_dir)

# =========================== BEGIN PROGRAM ================================

if __name__=="__main__" and debug_mode<4:	
    #### Check whether everything went well (no time exceeded)
    execution_success = True
    
    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = os.path.abspath(argv[2]);
        
    vprint( verbose,  "Using input_dir: " + input_dir)
    vprint( verbose,  "Using output_dir: " + output_dir)
        
    # Move old results and create a new output directory 
    if not(running_on_codalab) and save_previous_results:
        data_io.mvdir(output_dir, output_dir+'_'+the_date) 
    data_io.mkdir(output_dir) 
    
    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    # Overwrite the "natural" order
    
    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_io(input_dir, output_dir)
        print('\n****** Sample code version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')        	
        data_io.write_list(datanames)      
        datanames = [] # Do not proceed with learning and testing
        
    # ==================== @RESULT SUBMISSION (KEEP THIS) =====================
    # Always keep this code to enable result submission of pre-calculated results
    # deposited in the res/ subdirectory.
    if len(datanames)>0:
        vprint( verbose,  "************************************************************************")
        vprint( verbose,  "****** Attempting to copy files (from res/) for RESULT submission ******")
        vprint( verbose,  "************************************************************************")
        datanames = data_io.copy_results(datanames, res_dir, output_dir, verbose) # DO NOT REMOVE!
        if not datanames: 
            vprint( verbose,  "[+] Results copied to output directory, no model trained/tested")
        else:
            vprint( verbose, "======== Some missing results on current datasets!")
            vprint( verbose, "======== Proceeding to train/test:\n")
    # =================== End @RESULT SUBMISSION (KEEP THIS) ==================

    # ================ @CODE SUBMISSION  ================= 
    overall_time_budget = 0
    time_left_over = 0
    for basename in datanames: # Loop over datasets
        
        vprint( verbose,  "************************************************")
        vprint( verbose,  "******** Processing dataset " + basename.capitalize() + " ********")
        vprint( verbose,  "************************************************")
        
        # ======== Learning on a time budget:
        # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
        start = time.time()
        
        # ======== Creating a data object with data, informations about it
        vprint( verbose,  "========= Reading and converting data ==========")
        D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, max_samples=max_samples, verbose=verbose)
        print D
        vprint( verbose,  "[+] Size of uploaded data  %5.2f bytes" % data_io.total_size(D))
        
        # ======== Keeping track of time
        if debug_mode<1:
            time_budget = D.info['time_budget']        # <== HERE IS THE TIME BUDGET!
        else:
            time_budget = max_time
        overall_time_budget = overall_time_budget + time_budget
        vprint( verbose,  "[+] Cumulated time budget (all tasks so far)  %5.2f sec" % (overall_time_budget))
        # We do not add the time left over form previous dataset: time_budget += time_left_over
        vprint( verbose,  "[+] Time budget for this task %5.2f sec" % time_budget)
        time_spent = time.time() - start
        vprint( verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
        if time_spent >= time_budget:
            vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue
        
        # ========= Creating a model, knowing its assigned task from D.info['task'].
        # The model can also select its hyper-parameters based on other elements of info.  
        vprint( verbose,  "======== Creating model ==========")
        M = Classifier()
        
        # ========= Reload trained model if it exists.
        vprint( verbose,  "**********************************************************************")
        vprint( verbose,  "****** Attempting to reload model (from res/) to avoid training ******")
        vprint( verbose,  "**********************************************************************")
        you_must_train=1
        modelname = os.path.join(res_dir,basename)
        if os.path.isfile(modelname + '_model.pickle'):
            M = M.load(modelname)
            you_must_train=0
            vprint( verbose,  "[+] Model reloaded, no need to train!")
            
        # ========= Train if needed only.
        if you_must_train:
            vprint( verbose, "======== Trained model not found, proceeding to train!")
            start = time.time() 
            M.fit(D.data['X_train'], D.data['Y_train']) 
            vprint( verbose,  "[+] Fitting success, time spent so far %5.2f sec" % (time.time() - start))
            # Save model
            # ----------
            if save_model:
                outname = os.path.join(output_dir, basename)
                vprint( verbose, "======== Saving model to: " + output_dir)
                M.save(outname)
                vprint( verbose,  "[+] Success!")
            
            # Make predictions
            # -----------------
            Y_valid = M.predict_proba(D.data['X_valid'])
            Y_test = M.predict_proba(D.data['X_test'])                         
            vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
            # Write results
            # -------------
            if overwrite_output:
                filename_valid = basename + '_valid.predict'                
                filename_test = basename + '_test.predict'
            else:
                filename_valid = basename + '_valid_' + the_date + '.predict'                
                filename_test = basename + '_test_' + the_date + '.predict' 
            vprint( verbose, "======== Saving results to: " + output_dir)
            data_io.write(os.path.join(output_dir,filename_valid), Y_valid)
            data_io.write(os.path.join(output_dir,filename_test), Y_test)
            vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
            time_spent = time.time() - start 
            time_left_over = time_budget - time_spent
            vprint( verbose,  "[+] End cycle, time left %5.2f sec" % time_left_over)
            if time_left_over<=0: break
        # Clean up
        del D
        del M
        gc.collect()
            
    if zipme:
        vprint( verbose,  "========= Zipping this directory to prepare for submit ==============")
        data_io.zipdir(submission_filename + '.zip', ".")
        vprint( verbose,  "See: " + submission_filename + '.zip')
    	
    overall_time_spent = time.time() - overall_start
    if execution_success:
        vprint( verbose,  "[+] Done")
        vprint( verbose,  "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % overall_time_budget)
    else:
        vprint( verbose,  "[-] Done, but some tasks aborted because time limit exceeded")
        vprint( verbose,  "[-] Overall time spent %5.2f sec " % overall_time_spent + " > Overall time budget %5.2f sec" % overall_time_budget)
              
    if running_on_codalab: 
        if execution_success:
            exit(0)
        else:
            exit(1)


