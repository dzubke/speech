stream_log_spec = np.empty((0,257)).astype(np.float32) 
     ...: with wave.open(audio_path, 'rb') as wf:  
     ...:         chunk_size = 256 
     ...:         audio_ring_buffer = deque(maxlen=2)   
     ...:         num_samples = wf.getnframes()//chunk_size  
     ...:         print("num frames: ", wf.getnframes())  
     ...:         print("num_samples: ", num_samples)  
     ...:         for i in range(num_samples):  
     ...:             if len(audio_ring_buffer) < 1:  
     ...:                 audio_ring_buffer.append(wf.readframes(chunk_size))  
     ...:             else:  
     ...:                 audio_ring_buffer.append(wf.readframes(chunk_size))  
     ...:                 buffer_list = list(audio_ring_buffer)  
     ...:                 numpy_buffer = np.concatenate(  
     ...:                         (np.frombuffer(buffer_list[0], np.int16),   
     ...:                         np.frombuffer(buffer_list[1], np.int16)))  
     ...:                 log_spec_step = log_specgram_from_data(numpy_buffer, samp_rate=16000, window_size=32
     ...: , step_size=16) 
     ...:                 stream_log_spec = np.concatenate((stream_log_spec, log_spec_step), axis=0) 