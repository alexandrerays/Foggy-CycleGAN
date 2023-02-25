import os
import sys
import tensorflow as tf
from IPython.display import clear_output
from matplotlib import pyplot as plt

from lib.train import Trainer
from lib.plot import plot_clear2fog_intensity, plot_clear2fog_intensity_v2
from lib.tools import create_dir
from lib.dataset import DatasetInitializer
from lib.models import ModelsBuilder
import argparse
import numpy as np
import io

BATCH_SIZE = 1
IMG_WIDTH = 1024
IMG_HEIGHT = 1024
OUTPUT_CHANNELS = 3
INTENSITY = 0.9

# Config paths
weights_path = './weights/'
input_path = './runs/input/'
output_path = './runs/output/'

use_transmission_map = False #@param{type: "boolean"}
use_gauss_filter = False #@param{type: "boolean"}
use_resize_conv = False #@param{type: "boolean"}

datasetInit = DatasetInitializer(image_height=IMG_HEIGHT, image_width=IMG_WIDTH)
datasetInit.dataset_path = './dataset/'
models_builder = ModelsBuilder(image_height=IMG_HEIGHT, image_width=IMG_WIDTH, normalized_input=True)

generator_clear2fog = models_builder.build_generator(use_transmission_map=use_transmission_map,
                                                     use_gauss_filter=use_gauss_filter,
                                                     use_resize_conv=use_resize_conv)

generator_fog2clear = models_builder.build_generator(use_transmission_map=False)

use_intensity_for_fog_discriminator = False #@param{type: "boolean"}
discriminator_fog = models_builder.build_discriminator(use_intensity=use_intensity_for_fog_discriminator)
discriminator_clear = models_builder.build_discriminator(use_intensity=False)

trainer = Trainer(generator_clear2fog, generator_fog2clear,
                 discriminator_fog, discriminator_clear)

trainer.configure_checkpoint(weights_path = weights_path, load_optimizers=False)

trainer.load_config()

# loop through all files in the folder
for filename in os.listdir(input_path):
    # check if the file is a PNG file
    if filename.endswith('.png'):
        filename_with_path = os.path.join(input_path, filename)

        image_clear = tf.io.decode_png(tf.io.read_file(filename_with_path), channels=3)

        # Apply fog
        image_clear, _ = datasetInit.preprocess_image_test(image_clear, 0)
        fig = plot_clear2fog_intensity_v2(generator_clear2fog, image_clear, INTENSITY, normalized_input=True)

        # Save Images
        fig.set_size_inches(7, 7)
        fig_name_resized = os.path.join(output_path, "intensity_{:0.2f}_{}".format(INTENSITY, filename))
        fig.savefig(fig_name_resized, bbox_inches='tight', pad_inches=0, dpi=500)
        plt.close(fig)


