import os

img_shape = (128,128,1)
batch_size = 64
epochs = 200

base_output = 'output'
model_path = os.path.sep.join([base_output, "siamese_model"])
plot_path = os.path.sep.join([base_output, "plot.png"])
