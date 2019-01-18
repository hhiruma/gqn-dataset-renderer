import argparse
import numpy as np
import os
import random
import cv2
import math

def main():
    img_data_list = []
    vp_data_list = []

    for scene in os.listdir(args.input_directory):

        images_dir_path = os.path.join(args.input_directory, scene, "images")
        viewpoints_dir_path = os.path.join(args.input_directory, scene, "viewpoints")
        image_path_list = os.listdir(images_dir_path)
        viewpoint_path_list = os.listdir(viewpoints_dir_path)

        img_data_list_per_scene = []
        vp_data_list_per_scene = []

        for i in random.sample(range(len(image_path_list)), args.num_views_per_scene):
            img = cv2.imread(os.path.join(images_dir_path, image_path_list[i]))
            clipped_img = img[:,int((img.shape[1]-img.shape[0])/2):int(((img.shape[1]-img.shape[0])/2 + img.shape[0])),:]
            scaled_img = cv2.resize(clipped_img, dsize=(64,64))
            img_data_list_per_scene.append(scaled_img)

            with open(os.path.join(viewpoints_dir_path, viewpoint_path_list[i])) as f:
                line = f.read()
                vp = [float(s) for s in line.split(',')]
                viewpoint = [vp[0], vp[1], vp[2], math.sin(vp[3]), math.cos(vp[3]), math.sin(vp[4]), math.cos(vp[4])]

                vp_data_list_per_scene.append(viewpoint)

        img_data_list.append(img_data_list_per_scene)
        vp_data_list.append(vp_data_list_per_scene)

    if not os.path.exists(os.path.join(args.output_directory,"images")):
        os.mkdir(os.path.join(args.output_directory,"images"))
    if not os.path.exists(os.path.join(args.output_directory,"viewpoints")):
        os.mkdir(os.path.join(args.output_directory,"viewpoints"))

    np.save(os.path.join(args.output_directory,"images","data"),img_data_list)
    np.save(os.path.join(args.output_directory,"viewpoints","data"),vp_data_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-directory",
        "-in",
        type=str)
    parser.add_argument("--num-views-per-scene", "-k", type=int, default=15)
    parser.add_argument(
        "--output-directory",
        "-out",
        type=str,
        default="dataset_shepard_matzler_train")
    args = parser.parse_args()
    main()