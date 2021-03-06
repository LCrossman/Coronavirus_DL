# Coronavirus_DL

<h3> **  Installation  **</h3>

Requires Python 3.5 and above<br>
Prediction requires tensorflow 2.1.0 with Keras</br>
<br>
This is best installed <i>via</i> pip<br>
<br> pip install tensorflow</br>
Or tensorflow can be run in a docker container (see tensorflow website for more details)

Tensorflow is tested and supported on the following 64-bit systems:<br>
<br>
Python 3.5-3.7<br>
Ubuntu 16.04 or later<br>
Windows 7 or later<br>
MacOS 10.12.6 (Sierra) or later (no GPU support)<br>
Raspbian 9.0 or later<br>
<br>
<br>
Files containing computer simulated (DL) sequences:<br>
1. spike_from_random_1.fasta<br>
100 entirely simulated coronavirus spike protein sequences generated using seed texts with 16 amino acids selected at random <br>from the start of each protein in the training set.<br>
2. spike_from_sars_0.5.fasta<br>
Entirely simulated coronavirus spike protein sequences generated using a seed text of 64 amino acids from the start of SARS-CoV-2 spike protein.<br>
3. spike_1000_sars_0.5.fasta<br>
1000 simulated coronavirus spike protein sequences with a seed text of 64 amino acids from the start of SARS-CoV-2 spike<br>
<br><br>
<h2><b> To Predict:</b></h2><br>
Place the prediction model files (model.h5 and model.json) in the same directory<br>
To predict using random amino acids from the training set place seeds.txt in the same directory<br>

Run the prediction with the following options:<br>
<br>
  lengths = the length of sequence you want to generate<br><br>
  seqs = the number of separate sequences of length [lengths] you want to generate<br><br>
  outfile = the name of the file to save the output<br><br>
  random = True if you wish to use a random 16 amino acids as seed text (also require seeds.txt to generate this)<br>
           False or leave blank if you wish to use 64 amino acids from SARS-CoV-2 as seed text.<br><br>
  temperature = scaling parameter between 0 and 1, with higher values giving more surprising sequences and lower values<br>                  remaining more true to the original training set sequences<br>
<br>     
Example:<br>
     <b>python spike_sequence_generation.py --outfile tester --random True --lengths 1400 --seqs 10</b><br>

