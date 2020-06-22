#!/bin/bash
# copies files in a set format and directory structure to a destination directory on the local machine. i
# this is not a general tool. If all the filenames aren't the same or in the right directory, this won't work.
# Note: this command needs to be run on a local machine. 
# commmand structure: bash scp_debug_files.sh <$1=phoneme_number> <$2=training_run_base_filename> <$3=destintation_dir> 

# copy the output file
gcloud compute scp  dzubke@phoneme-$1:/home/dzubke/awni_speech/speech/$2.out $3

# copy the log file
gcloud compute scp  dzubke@phoneme-$1:/home/dzubke/awni_speech/speech/logs/$2.log $3

# copy the plot file
gcloud compute scp  dzubke@phoneme-$1:/home/dzubke/awni_speech/speech/plots/$2_bar.png $3

# copy the saved last batch file
gcloud compute scp  dzubke@phoneme-$1:/home/dzubke/awni_speech/speech/saved_batch/$2_batch.pickle $3

