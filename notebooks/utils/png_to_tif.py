
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pyvips


ROOT_DIR = '/tank/WSI_data/Ovarian_WSIs/UBC-OCEAN/'
TRAIN_DIR = '/tank/WSI_data/Ovarian_WSIs/UBC-OCEAN/train_images'

df = pd.read_csv(f"{ROOT_DIR}/train.csv")

# create a list of image_id from the filtered rows
image_id_list = df['image_id'].tolist()
# print(image_id_list)
png_list = [f"{i}.png" for i in image_id_list]

print(png_list)
print("Number of images: ",len(png_list))

# loop through the png_list
for png_file in png_list:
    # read a png file from the directory
    image = pyvips.Image.new_from_file(f"{TRAIN_DIR}/{png_file}")
    
    # write to a tiff file with some options
    image.tiffsave(f"/tank/WSI_data/Ovarian_WSIs/UBC-OCEAN/tifs/{png_file[:-4]}.tif", compression="jpeg", Q=95, tile=True, pyramid=True)



