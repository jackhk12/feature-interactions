import os
from matplotlib.cm import winter
import platform
import cv2  # For selecting AOIs
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
pd.set_option("mode.chained_assignment", None)
import matplotlib as mpl  # For plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import patches
import seaborn as sns
sns.set_context("paper")  # Set the context of the plots in seaborn
import tkinter as tk  # For GUI
from tkinter import filedialog
import logging  # For logging
import argparse  # For parsing arguments
from enum import Enum  # For enumerating the metrics

# Preparing the logger
logging.getLogger("defineAOIs")
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# What metrics are available?
class MetricClasses(Enum):
    all = 0
    hit_rate = 1
    first_contact = 2
    dwell_time = 3
    
class DataClasses(Enum):
    fixations = 0
    gaze = 1

# produces a GUI to prompt user to select the folder containing the fixations.csv, gaze.csv, sections.csv, and reference_image.jpeg ENDOW files
def get_path():
    root = tk.Tk()
    root.withdraw()
    msg = "Select the directory"
    arguments = {"title": msg}
    path = filedialog.askdirectory(**arguments)
    # check if the folder contains the required files
    if (
        not os.path.exists(os.path.join(path, "fixations.csv"))
        or not os.path.exists(os.path.join(path, "gaze.csv"))
        or not os.path.exists(os.path.join(path, "sections.csv"))
        or not os.path.exists(os.path.join(path, "reference_image.jpeg"))
    ):
        error = f"The selected folder does not contain a reference_image.jpeg, fixations.csv, gaze.csv or sections.csv files"
        logging.error(error)
        raise SystemExit(error)
    return path



def main():
    parser = argparse.ArgumentParser(description="Pupil Labs - AOI Annotation")
    parser.add_argument("--metric", default=MetricClasses.all, type=str)
    parser.add_argument("--input_path", default=None, type=str)
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument("--aois", default=None, type=str)
    parser.add_argument("--start", default="recording.begin", type=str)
    parser.add_argument("--end", default="recording.end", type=str)
    parser.add_argument("--type", default=DataClasses.fixations, type=str)
    parser.add_argument("-s", "--scatter", action="store_true")
    parser.set_defaults(scatter=True)
    args = parser.parse_args()


    # ac_

    # Report selected arguments
    logging.info("args: %s", args)
    if isinstance(args.metric, str):
        args.metric = MetricClasses[args.metric]
        logging.info("metric: %s", args.metric)
    # check if is string
    if isinstance(args.type, str):
        args.type = DataClasses[args.type]
        logging.info("type: %s", args.type)

    # If the reference image folder path is not provided or does not exist, ask the user to select one
    if args.input_path is None or not os.path.exists(args.input_path):
        args.input_path = get_path()
    # If the output path is not provided or does not exist, use the input path
    if args.output_path is None or not os.path.exists(args.output_path):
        args.output_path = args.input_path
    logging.info("Input path: %s", args.input_path)

    # Load the reference image
    reference_image_bgr = cv2.imread(
        os.path.join(args.input_path, "reference_image.jpeg")
    )

    # Convert the image to BGR for OpenCV
    reference_image = cv2.cvtColor(reference_image_bgr, cv2.COLOR_BGR2RGB)

    # If the AOIs are not provided, ask the user to select them
    if args.aois != None:
        # Load the AOIs from a csv file
        aois = pd.read_csv(args.aois)
    elif args.aois == None:
        # if there is an aois.csv file in the input path, use it
        if os.path.exists(os.path.join(args.input_path, "aoi_ids.csv")):
            logging.info("AOIs already defined")
            args.aois = os.path.join(args.input_path, "aoi_ids.csv")
            aois = pd.read_csv(args.aois)
        else:
            # Resize the image before labelling AOIs makes the image stay in the screen boundaries
            scaling_factor = 0.25
            scaled_image = reference_image_bgr.copy()
            scaled_image = cv2.resize(
                scaled_image, dsize=None, fx=scaling_factor, fy=scaling_factor
            )
            # mark the AOIs
            scaled_aois = cv2.selectROIs("AOI Annotation", scaled_image)
            cv2.destroyAllWindows()

            # Scale back the position of AOIs
            aois = scaled_aois / scaling_factor

            # Save the AOIs and their values to a pandas DataFrame
            aois = pd.DataFrame(aois, columns=["x", "y", "width", "height"])

            # Save the AOIs to a csv file
            aois.to_csv(args.output_path + "/aoi_ids.csv", index=False)
    logging.info("Areas of interest:")
    logging.info(aois)









    

    # Plot the AOIs
    plot_color_patches(reference_image, aois, pd.Series(aois.index), plt.gca(), args)

    # Load the sections file and the fixations file onto pandas DataFrames
    logging.info("Loading files ...")
    sections_df = pd.read_csv(args.input_path + "/sections.csv")
    logging.info(sections_df["start event name"].unique())
    logging.info(sections_df["end event name"].unique())

    #create a new dataframe to parse the fixations file
    fixations_df = pd.read_csv(args.input_path + "/fixations.csv")
    logging.info("A total of %d fixations were found", len(fixations_df))


    #create a new dataframe to parse the gaze file
    gaze_df = pd.read_csv(args.input_path + "/gaze.csv")
    logging.info("A total of %d gaze points were found", len(gaze_df))

    # Make data fixations or gaze, depending on the selected type
    data_df = fixations_df if args.type == DataClasses.fixations else gaze_df
    field_detected = (
        "fixation detected in reference image"
        if args.type == DataClasses.fixations
        else "gaze detected in reference image"
    )
    # filter for fixations that are in the reference image and check which AOI they are in
    data = data_df[data_df[field_detected]]

    for row in aois.itertuples():
        data_in_aoi = data.copy()
        data_in_aoi = data.loc[
            check_in_rect(data, [row.x, row.y, row.width, row.height], args)
        ]
        data.loc[data_in_aoi.index, "AOI"] = row.Index

    logging.info(f"A total of %d {args.type} points were detected in AOIs", len(data))

    # AOIs that have never been gazed at do not show up in the fixations data
    # so we need to set them to 0 manually
    hits = data.groupby(["recording id", "AOI"]).size() > 0
    hit_rate = hits.groupby("AOI").sum() / data["recording id"].nunique() * 100
    for aoi_id in range(len(aois)):
        if not aoi_id in hit_rate.index:
            hit_rate.loc[aoi_id] = 0
    hit_rate.sort_index(inplace=True)
    logging.info("Hit rate per AOI:")
    logging.info(hit_rate.head())

    # Compute the time difference for the respective section
    sections_df.set_index("section id", inplace=True)
    for section_id, start_time in sections_df["section start time [ns]"].iteritems():
        data_indices = data.loc[data["section id"] == section_id].index
        logging.info(
            "The section {} starts at {} and has {} points".format(
                section_id,
                start_time,
                len(data_indices),
            )
        )
        field_ts = (
            "start timestamp [ns]"
            if args.type == DataClasses.fixations
            else "timestamp [ns]"
        )
        data.loc[data_indices, "aligned timestamp [s]"] = (
            data.loc[data_indices, field_ts] - start_time
        ) / 1e9
    first_contact = data.groupby(["section id", "AOI"])["aligned timestamp [s]"].min()
    first_contact = first_contact.groupby("AOI").mean()
    logging.info(first_contact)

    # Compute the dwell time for the respective AOI
    if args.type == DataClasses.fixations:
        dwell_time = data.groupby(["recording id", "AOI"])["duration [ms]"].sum()
        dwell_time = dwell_time.groupby("AOI").mean()
        dwell_time /= 1000
        logging.info(dwell_time.head())


### BEGIN PLOT 1 ###

    if args.type == DataClasses.fixations:
        if args.metric == MetricClasses.hit_rate or args.metric == MetricClasses.all:
            fig, ax = plt.subplots(1, 2, figsize=(18, 6))
            ax[0].set_xlabel("AOI ID")
            ax[0].set_ylabel("Hit Rate [% of testers]")

            min_value = min(hit_rate)
            max_value = max(hit_rate)
            
            norm = colors.Normalize(vmin=min_value, vmax=max_value)
            cmap = mpl.cm.get_cmap("winter")
            #bar_colors = cmap(hit_rate / hit_rate.max())
            bar_colors = cmap(norm(hit_rate))

    
            for i, v in enumerate(hit_rate.values):
                ax[0].text(i, v + 1, format(v, '.2f'), color='black', ha="center")
                ax[0].grid('on', color='lightgray', linestyle='dashed')
                ax[0].grid(zorder=0)
                ax[0].bar([i], v, width=0.3, align='center', color=bar_colors[i], edgecolor='black', zorder=3)
                ax[0].spines['right'].set_color((.8,.8,.8))
                ax[0].spines['top'].set_color((.8,.8,.8))
    


            plot_color_patches(
                reference_image,
                aois,
                hit_rate,
                ax[1],
                args,
                colorbar=True,
                unit_label="Hit Rate [%]",
                data=data,
            )

        fig.suptitle(f"Hit Rate - {args.type}")
    else:
        logging.info("Dwell time is only available for fixations data")
   

#### END PLOT 1 ###

### BEGIN PLOT 2 ###

    if args.metric == MetricClasses.first_contact or args.metric == MetricClasses.all:
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        ax[0].set_xlabel("AOI ID")
        ax[0].set_ylabel("Time to first contact [s]")

        min_value = min(first_contact)
        max_value = max(first_contact)
            
        norm = colors.Normalize(vmin=min_value, vmax=max_value)
        cmap = mpl.cm.get_cmap("winter")
        bar_colors = cmap(norm(first_contact))

        #cmap = mpl.cm.get_cmap("turbo", len(hit_rate))
        #bar_colors = cmap(range(len(hit_rate)))

        for i, v in enumerate(first_contact):
            ax[0].text(i, v + 0.1, "{:.2f}".format(v), color='black', ha="center")
            ax[0].grid('on', color='lightgray', linestyle='dashed')
            ax[0].grid(zorder=0)
            ax[0].bar([i], v, width=0.3, align='center', color=bar_colors[i], edgecolor='black', zorder=3)
            ax[0].spines['right'].set_color((.8,.8,.8))
            ax[0].spines['top'].set_color((.8,.8,.8))

        plot_color_patches(
            reference_image,
            aois,
            first_contact,
            ax[1],
            args,
            colorbar=True,
            unit_label="Time to first contact [s]",
            data=data,
        )
    fig.suptitle(f"First Contact - {args.type}")

#### END PLOT 2 ###

### BEGIN PLOT 3 ###

    if args.type == DataClasses.fixations:
        if args.metric == MetricClasses.dwell_time or args.metric == MetricClasses.all:
            fig, ax = plt.subplots(1, 2, figsize=(18, 6))
            ax[0].set_xlabel("AOI ID")
            ax[0].set_ylabel("Dwell Time [s]")

            min_value = min(dwell_time)
            max_value = max(dwell_time)
            
            norm = colors.Normalize(vmin=min_value, vmax=max_value)
            cmap = mpl.cm.get_cmap("winter")
            #bar_colors = cmap(hit_rate / hit_rate.max())
            bar_colors = cmap(norm(dwell_time))

            for i, v in enumerate(dwell_time):
                ax[0].text(i, v + 0.01, "{:.2f}".format(v), color='black', ha="center")
                ax[0].grid('on', color='lightgray', linestyle='dashed')
                ax[0].grid(zorder=0)
                ax[0].bar([i], v, width=0.3, align='center', color=bar_colors[i], edgecolor='black', zorder=3)
                ax[0].spines['right'].set_color((.8,.8,.8))
                ax[0].spines['top'].set_color((.8,.8,.8))

            plot_color_patches(
                reference_image,
                aois,
                dwell_time,
                ax[1],
                args,
                colorbar=True,
                unit_label="Dwell Time [s]",
                data=data,
            )
        fig.suptitle(f"Dwell Time - {args.type}")
    else:
        logging.info("Dwell time is only available for fixations data")

pd.reset_option("mode.chained_assignment")
logging.info("Done")

### END PLOT 3 ###


def check_in_rect(data, rectangle_coordinates, args):
    rect_x, rect_y, rect_width, rect_height = rectangle_coordinates
    if args.type == DataClasses.fixations:
        fieldx = "fixation x [px]"
        fieldy = "fixation y [px]"
    elif args.type == DataClasses.gaze:
        fieldx = "gaze position in reference image x [px]"
        fieldy = "gaze position in reference image y [px]"
    x_hit = data[fieldx].between(rect_x, rect_x + rect_width)
    y_hit = data[fieldy].between(rect_y, rect_y + rect_height)
    return x_hit & y_hit


def plot_color_patches(
    image,
    aoi_positions,
    values,
    ax,
    args,
    colorbar=False,
    unit_label="",
    data=None,
):
    
    ax.imshow(image)
    ax.axis("off")
    
    # normalize patch values
    values_normed = values.astype(np.float32)
    values_normed -= values_normed.min()
    values_normed /= values_normed.max()

    colors = mpl.cm.get_cmap("winter")
    cmap = mpl.cm.get_cmap("winter")

    for aoi_id, aoi_val in values_normed.iteritems():
        aoi_id = int(aoi_id)
        aoi = [
            aoi_positions.x[aoi_id],
            aoi_positions.y[aoi_id],
            aoi_positions.width[aoi_id],
            aoi_positions.height[aoi_id],
        ]
        ax.add_patch(
            patches.Rectangle(
                aoi,
                *aoi[2:],
                #facecolor=colors(aoi_val, alpha =0.3),
                #edgecolor=colors(aoi_val, alpha=2.0),

                facecolor=cmap(aoi_val, alpha =0.3),
                edgecolor=cmap(aoi_val, alpha=2.0),
                linewidth=1,
            )
    
        )
        
        # line below enables (or disables, if commented out) AOI IDs in the corner of each AOI
        ax.text(aoi[0] + 20, aoi[1] + 120, f"{aoi_id}", color="white")

    
    if colorbar:
        norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colors), ax=ax)
        cb.set_label(unit_label)


    if data is not None and args.scatter:
        if args.type == DataClasses.fixations:
            field0 = "fixation detected in reference image"
            field1 = "fixation x [px]"
            field2 = "fixation y [px]"
        elif args.type == DataClasses.gaze:
            field0 = "gaze detected in reference image"
            field1 = "gaze position in reference image x [px]"
            field2 = "gaze position in reference image y [px]"
        data_in = data[data[field0] == True]
        ax.scatter(data_in[field1], data_in[field2], s=15, color="red", alpha=0.3)

    plt.show()
    return ax


if __name__ == "__main__":
    main()