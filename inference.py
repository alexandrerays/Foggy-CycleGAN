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
from PIL import Image
import numpy as np
import io

step = 0.05

BATCH_SIZE = 1
IMG_WIDTH = 1024
IMG_HEIGHT = 1024
OUTPUT_CHANNELS = 3

weights_path = "./weights/"

use_transmission_map = False #@param{type: "boolean"}
use_gauss_filter = False #@param{type: "boolean"}
use_resize_conv = False #@param{type: "boolean"}

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--case", required=True, help="path to input image")
args = vars(ap.parse_args())
test_case = args["case"]

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

intensity_path = f'./runs/output/{test_case}/'
create_dir(intensity_path)
file_path = f'./runs/input/{test_case}.png'

image_clear = tf.io.decode_png(tf.io.read_file(file_path), channels=3)

original_dimensions = (image_clear.shape[1], image_clear.shape[0])
print(original_dimensions)

image_clear, _ = datasetInit.preprocess_image_test(image_clear, 0)


fig = plot_clear2fog_intensity_v2(generator_clear2fog, image_clear, 0.9, normalized_input=True)
fig.set_size_inches(10, 10)
#ind = ind+20
fig_name_resized = os.path.join(intensity_path, "intensity_{:0.2f}_resized.jpg".format(0.9))
fig_name = os.path.join(intensity_path, "intensity_{:0.2f}.jpg".format(0.9))
fig.savefig(fig_name_resized, bbox_inches='tight', pad_inches=0, dpi=200)
plt.close(fig)

image = Image.open(fig_name_resized)
new_image = image.resize(original_dimensions)
new_image.save(fig_name)

