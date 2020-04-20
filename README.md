# Coronavirus_DL

**  Installation  **

Requires Python 3.5 and above
Prediction requires tensorflow 2.1.0 with Keras backend</br>
<br>
This is best installed <i>via</i> conda<br>
<br>
Place the prediction model files (model.h5 and model.json) in the same directory<br>
Run the prediction with the following options:<br>

  lengths = the length of sequence you want to generate<br>
  seqs = the number of separate sequences of length [lengths] you want to generate<br>
  outfile = the name of the file to save the output<br>
  random = True if you wish to use a random 16 amino acids as seed text (also require seeds.txt to generate this)<br>
           False or leave blank if you wish to use 64 amino acids from SARS-CoV-2 as seed text.<br>
<br>
<br>           
Example:<br>
     python spike_sequence_generation.py --outfile tester --random True --lengths 25 --seqs 5<br>

