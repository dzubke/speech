# this file is a basic script and should be made more robust if it will be used often

# standard libraries
import os

FILENAMES = [
    'common_voice_en_17298604.mp3', 
    'common_voice_en_18364975.mp3',
    'common_voice_en_17736843.mp3',
    'common_voice_en_18377509.mp3',
    'common_voice_en_1531516.mp3',
    'common_voice_en_17356677.mp3',
    'common_voice_en_20047839.mp3',
    'common_voice_en_17990270.mp3',
    'common_voice_en_18034194.mp3',
    'common_voice_en_18385164.mp3',
    'common_voice_en_18958067.mp3',
    'common_voice_en_18864859.mp3',
    'common_voice_en_514043.mp3',
    'common_voice_en_508297.mp3',
    'common_voice_en_508299.mp3',
    'common_voice_en_17831791.mp3',
    'common_voice_en_17831799.mp3'
]

def main():
    """
    transfer filese in FILENAMES from the src_dir to the dst_dir
    """
    gcloud_command = "gcloud compute scp {src_path} {dst_path}"
    src_dir = "dzubke@phoneme-2:~/awni_speech/data/common-voice/clips"
    dst_dir = "/Users/dustin/CS/consulting/firstlayerai/data/dataset_samples/common-voice_canada"
    extension = "mp3"
    for filename in FILENAMES:
        wv_filename = os.path.splitext(filename)[0] + os.extsep + extension
        src_path = os.path.join(src_dir, wv_filename)
        dst_path = os.path.join(dst_dir, wv_filename)
        os.system(gcloud_command.format(src_path=src_path, dst_path=dst_path))


if __name__ == "__main__":
    main()