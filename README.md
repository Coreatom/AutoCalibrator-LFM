# AutoCalibrator-LFM

## Self-undistort
This code contains a method to undistort images of the acrylic board.
## project architecture
``` bash
├── code
│   ├── main.py         # Main script for the project
│   └── realign.py      # Script for realignment operations
├── config
│   ├── environment.yml # Python environment
│   └── test.json       # Configuration file for testing
├── data
│   ├── param           # Parameter files
│   ├── temp            # Temporary data generated during processing
│   └── image
│       ├── raw         # Raw image data
│       └── result      # Processed image results
└── figure              # Output figures and plots
```
## How to use it?

1. Paste the raw image of the acrylic board (file name: `white488_20X_S1_C2_0.tiff`) into the path `data/image/raw`.

2. In this project, all paths use relative paths, so you don't need to edit the file paths. If you do wish to edit them, you can open `config/test.json` and check the `"image_path"` and `"save_path"` fields. The `"image_path"` should point to the location where the raw image of the acrylic board is saved. You can also verify the config file path in `code/main.py`.

3. Please run `main.py`.

4. You will generate a `.pkl` file containing the parameters, which will be saved in the `data/param` folder. An undistorted image of the acrylic board will also be created in the same folder. Additionally, some data for `show.py` will be generated in the `data/temp` folder.

5. In the `data/param` folder, you will find a file named `undistort_params_dict_points_{date}.pkl`. This file is intended for the `realign_panorama` project (note that the `realign_panorama` project is not included in this self-undistort project). You can copy it to `realign_panorama/reconstruction/source/realign` and delete the files `undistort_params_dict_points_240620.pkl`, `sub_x_undistorted.npy`, and `sub_y_undistorted.npy`. Then, run the `realign_panorama` project, and the new undistort parameters will be used to realign the images of biological samples.

6. If you would like to see the specific algorithm details, please open `code/realign.py` and read the comments at the beginning of each function.

## Configuration Parameters Explanation

This section provides detailed explanations of the parameters defined in the configuration file `test.json`.

1. **`image_path`**  
   Specifies the path to the input image file. It should be a image of acrylic board.

2. **`save_path`**  
   Indicates where to save the results.

3. **`crop_H`** and **`crop_W`**  
   Height and width of the cropping region in pixels. Must be multiples of 15.

4. **`start_H`** and **`start_W`**  
   Starting coordinates for cropping from the top and left of the image. These values should not be too small, as they may lead to cropping into areas where the microlenses cannot be accurately identified.

5. **`block_H`** and **`block_W`**  
   Number of blocks for processing in height and width directions. Should be even, and not too small.

6. **`step`**  
   Step size used in the processing algorithm. Must be multiples of 15. Normally, it can be 300. 

7. **`max_threshold_rate`** and **`min_threshold_rate`**  
   Maximum and minimum threshold rates for pixel intensity.

8. **`step_threshold_rate`**  
   Incremental step for adjusting threshold rates.

9. **`half_size`**  
   Half size of the image block for identifying microlens.

10. **`min_centroids_number`**, **`min_point_distance`** and **`near_points_range`**  
    Some params in the microlen identification. 
