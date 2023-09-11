# feature-interactions

Ocular Behavior Visualizer Script for User Eye-Tracking Research

![Python Version](https://img.shields.io/badge/Python-3.6%2B-brightgreen.svg)

This my upgraded iteration Pupil Labs AOI (Area of Interest) Annotation Tool is a Python script for annotating and analyzing eye-tracking data. This tool is designed to work with eye-tracking data from Pupil Labs eye trackers and allows you to define AOIs on a reference image, calculate metrics related to these AOIs, and visualize the results.

## Features

- **AOI Annotation**: Define Areas of Interest (AOIs) on a reference image to specify regions of interest for eye-tracking analysis.

- **Metrics Calculation**: Calculate various metrics related to eye-tracking data, including Hit Rate, Time to First Contact, and Dwell Time.

- **Data Visualization**: Visualize eye-tracking data, AOIs, and metric results on the reference image.

- **Command-Line Interface**: Use a command-line interface to customize and control the analysis process.


## My Upgrades


- **AOI Heatmap Correspondence**: Defined AOIs now match their respective quantitative heatmap integer. The original script returned a pre-metric AOI field map without bounding box (fill/border color) differentiation, so I made each AOI field correspond to the heatmap colorbar for the respective metric. The pre-metric heatmap averages out each AOI field across a static colorbar for rapid reference. 

- **Bar Graph - AOI Matching**: AOIs represented on returned metric visualizers now match their respective AOI on the reference image HUD. The original script did not match these, and operated on an undefined colorbar, which inadvertently created difficulties in analytics synthesis and presentation. My upgrades allow for rapid reference and greater visualizer cohesion. 

- **Tkinter GUI File Selection**: Raw data can now be selected and with the help of a file navigation GUI. The original script required users to manually specify file location in-script, which became a hassle when dealing with inherent file naming schema variance during high-tempo research cycles. My upgrade allows for simple parent folder selection of the required .csv files (details below), dramatically streamlining script execution and script/dataset/filename location changes.

- **Fixation Display Visualizer Toggle**: Tester fixation density (excluding saccades and similar scan behavior) was not available in the original script. Sequestering this information from the data visualizer, which contained unlogged edge-case fixations and datapoints in regions outside the user-drawn bounding boxes, painted an incomplete picture of user behavior and ocular interaction metrics. I enabled these fixation visualizers in the form of translucent red circles, which overlap or stray to create an overlay scatterplot of fixation data. 

## Prerequisites

- Python 3.6 or higher

## Usage

1. Clone this repository or download the script.

2. Install the required Python packages using pip:

   ```shell
   pip install matplotlib pandas opencv-python-headless seaborn
   ```

3. After running the proper change directory commands, running the script will prompt you with a tkinter GUI to select the folder containing the appropriate fixations.csv, gaze.csv, sections.csv, and reference_image.jpeg.
   
4. Successful file selection will open the reference image, which will allow the user to drag-and-draw bounding boxes around features of interest.

5. After a box has been drawn, press the spacebar to confirm AOI addition. When all AOIs have been drawn, hit "esc" to generate metrics. 

## Configuration

You can customize the analysis and metrics calculation by modifying the script's command-line arguments. Refer to the script's documentation for a list of available options.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This script was developed for use with Pupil Labs eye-tracking data.
Note: This tool is intended for research and analysis purposes. Make sure you have the necessary permissions to use eye-tracking data for your specific application.

Feel free to contribute to this project or report any issues you encounter.
