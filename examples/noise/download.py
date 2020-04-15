# this file  downloads the source file to create a collection
# of long-duration background noise files

# standard library
import json
import os
import time
import argparse
# third party libraries
import requests
import tqdm


def main(download_dir:str):
    """
    collects the sound_id, downloads them from a parses GET requet to freesound.org,
    and writes them to download_dir
    """
    sound_ids = collect_soundid()
    get_and_write(sound_ids, download_dir)


def collect_soundid(credit_file:str="./FREESOUNDCREDITS.txt")->list:
    """
    parses the credits from the UrbanSound dataset in the file FREESOUNDCREDITS.txt
    and returns the sound_id values as list
    """

    skip_lines = 6  # number of lines to skip
    id_list = list()
    with open(credit_file, 'r') as fid:
        for index, line in enumerate(fid):
            # skip the first header lines
            if index <= skip_lines:
                continue
            # format of parsed_line is [soundid, "by", username]
            parsed_line = line.strip().split()
            soundid = parsed_line[0]
            #check the value is  what we expect
            assert int(soundid), "sounid is not string of integer"
            id_list.append(soundid)
    return id_list


def get_and_write(sound_ids:list, download_dir:str):
    """
    sends a GET request to the freesound API to retrieve information about each value in 
    sound_ids and writes the low-quality audio identified by sound_id to the download_dir
    """
    # replace APIKEY_FILE to file with freesound API key
    APIKEY_FILE = "/home/dzubke/awni_speech/freesound_apikey.txt"
    with open(APIKEY_FILE, 'r') as fid:
        API_KEY = fid.readline().strip()
    print(f"api key loaded:{API_KEY}")
    
    GET_TEMPLATE = "https://freesound.org/apiv2/sounds/{sound_id}/?token={API_KEY}"
    min_duration = 20   # min file duration to filter upon
    count_filesnotfound = 0
    count_shortfiles = 0
    wait_time = 0.100    # wait 100 ms between api calls

    print("Processing sound_id list...")
    for sound_id in tqdm.tqdm(sound_ids):
        response = requests.get(GET_TEMPLATE.format(sound_id=sound_id, API_KEY=API_KEY))
        properties = json.loads(response.text)
        if len(properties)==1: 
            print(f"sound id: {sound_id} not found on freesound.org")
            count_filesnotfound += 1
            continue     # sound_id was not found, skip to next example
        if properties.get("duration") > min_duration:
            # retrieve the url for the low-quality mp3 version
            download_url =  properties.get("previews").get('preview-lq-mp3')
            # get the download_url file and write to download_dir
            download_response = requests.get(download_url)
            download_file = os.path.basename(download_url)
            download_file = os.path.join(download_dir, download_file)
            with open(download_file, 'wb') as fid:
                fid.write(download_response.content)
            time.sleep(wait_time)    # sleep to not overload API
        else: 
            count_shortfiles +=1

    print("finished processing")
    print(f"Number files not found: {count_filesnotfound}")
    print(f"Number files less than  {min_duration} sec: {count_shortfiles}")


if __name__=="__main__":

    parser = argparse.ArgumentParser(
            description="Downloading selection of full noise files from UrbanSound dataset")
    parser.add_argument("--download-dir", type=str,
        help="Directory where noise audio will be downloaded.")
    args = parser.parse_args()

    main(args.download_dir)

