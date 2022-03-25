# PassiveActiveGit
Passive Active Vision

Using the Potsdam Search Corpus data, found in DOI 10.17605/OSF.IO/JQ56S, and its corresponding raw eye tracking data. 

Parsing of the data was done in two steps:

1. Functions in script merging_data.py were combined in the script iterating_files.ipynb. 
For each participant we did:

  a) get all valid time stamps, i.e, filter the times such that all times in the raw data are in between the imagestart and imageend times in the corpus data.
  
  b) transform the data from pixels to dvas. This is done with a linear transformation. 
  
  c) associate each time stamp and trial to the correspoding image number, filter type and region, presence of target and expected location, according to the corpus data.
  
  d) by comparing time stamps with the corpus_data, we get which time stamps corresponds to saccades (=1) and which dont (=0). Add an id that corresponds to 3 digits of subjet number + 3 digits trial number + 2 digits fixation/saccade number. Also add information about invalid saccades/fixations, in invalid column where valid=0 and invalid=1. REMARK: identifier column always need to be imported as a string! 
  
  e) save the data in files '../separate_participant_data/Raw_Data/raw_subj_'+str(subject_number)+'.dat'
  
2. Functions in script replay_condition.py were combined in the script create_replay_general.ipynb. In here we always use the data from step 1. 
For each participant and each image we did:
  
  a) filtering data with a Savitzky-Golay filter. 
  
  b) upsampling the data to a frequency of 1440Hz. 
  
  c) adding contrast ramps to invalid fixations/saccades. This is called the continuous condition and therefore we add a contrast_cont column to the dataframe. This is done with the add_contrast_invalid function, that fits a Gaussian CDF for invalid saccades and fixations. We consider mean 50 and std = 50/3. The contrast ramp in this case is limited to 100ms down or up. For invalid intervals that last more than that, we keep the contrast at zero in the middle.
  
  d) adding contrast ramps to valid saccades. This is done by the function add_contrast_valid_apparent_case. The contrast ramp is calculated in steps: First we transform the position into position deltas, then we fit a Gumbel CDF to it, then we take the derivative of the fitted data and finally we invert and normalize the profile encoutered.
  
   Changes in x,y are also performed such that in the first half of the contrast ramp we have x,y coordinates fixed to what they were before the beggining of the saccade. In the seccond half, the x,y coordinates are fixed to the fixation after the saccade ends. This is called the apparent condition and therefore we add a contrast_app column to the dataframe. Also we add columns x/y_apparent, corresponding to the modified x,y coordinates. 
  
  e) the data from all images that were presented to a participant is combined in a final dataframe, which is then saved in files '../separate_participant_data/Final_Data/data_subj_'+str(subject_number)+'.dat'.
  
 
 !!Final data can be found in: https://osf.io/v2cdh/
