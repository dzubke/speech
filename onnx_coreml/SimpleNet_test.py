import coremltools
# Load the model
model = coremltools.models.MLModel('SimpleNet_2020-01-30.mlmodel')
# Visualize the model
model.visualize_spec()