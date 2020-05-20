import coremltools
import numpy as np
# Load the model
model = coremltools.models.MLModel('./coreml_models/resnet18.mlmodel')
# Visualize the model
input_dict = {'input': np.random.randn(1, 3, 224, 224).astype(np.float32)}
print(model.predict(input_dict))
