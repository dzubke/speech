from speech.utils import wave, noise_injector

def check_length(audio_path:str, noise_path:str, noise_level:float=0.5):
    audio_data, samp_rate = wave.array_from_wave(audio_path)
    audio_noise = noise_injector.inject_noise_sample(audio_data, samp_rate, noise_path, 
                    noise_level=noise_level, logger=None)
