"""
THIS CODE DOESN'T WORK IN ITS CURRENT FORM AS I CAN'T TRANSFER FILES
FROM A GCP INSTANCE TO MY LOCAL, SO I RUN THE FIRST LINES ON GCP TO GET
THE CHOSEN AUDIO FILES AND THEN RUN THE REMAINING CODE ON MY LOCAL.S

This code transfers a random selection of audio files from various datasets
to my local machine from a google cloud instance
"""
# standard libraries
import os
# third party libraries
import numpy as np
# project libraries
from speech.dataset_info import CommonvoiceDataset, TedliumDataset, Librispeech500Dataset


LOCAL_DIR = "/Users/dustin/CS/consulting/firstlayerai/data/dataset_samples"
GCP_USER_PROJECT = "dzubke@phoneme-2:"


def main():
    datasets = [CommonvoiceDataset(), TedliumDataset(), Librispeech500Dataset()]
    num_files = 50  # number of files to transfer

    for dataset in datasets:

        audio_files = dataset.get_audio_files()
        # selects num_file audio_files without replacement
        audio_choices = np.random.choice(audio_files, size=num_files, replace=False)
        dst_dir = os.path.join(LOCAL_DIR, dataset.dataset_name)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        cmd_str = "gcloud compute scp "+GCP_USER_PROJECT+"{audio_file} "+dst_dir

        for audio_file in audio_choices:
            os.system(cmd_str.format(audio_file=audio_file))



if __name__ == "__main__":
    main()



