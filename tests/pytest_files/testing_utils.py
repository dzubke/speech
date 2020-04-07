import os

def skip_files(dataset_name:str, audio_path:str)->bool:
    """
    if the audio path is in one of the noted files with errors, return True
    """
    valid_data_names = ["librispeech", "tedlium", "common-voice", "voxforge", "tatoeba"]
    if dataset_name not in valid_data_names: raise ValueError("Invalid dataset name")
    
    sets_with_errors = ["tatoeba"]
    # CK is directory name and min, max are the ranges of filenames
    tatoeba_errors = {"CK": {"min":6122903, "max": 6123834}}
    skip = False
    if dataset_name not in sets_with_errors:
        # jumping out of function to reduce operations
        return skip
    file_name, ext = os.path.splitext(os.path.basename(audio_path))
    dir_name = os.path.basename(os.path.dirname(audio_path))
    if dataset_name == "tatoeba":
        for tat_dir_name in tatoeba_errors.keys():
            if dir_name == tat_dir_name:
                if tatoeba_errors[tat_dir_name]["min"] <= int(file_name) <=tatoeba_errors[tat_dir_name]["max"]:
                    skip = True
    return skip     
