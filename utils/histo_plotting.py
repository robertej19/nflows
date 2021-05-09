
from utils import make_histos
import os, sys
from icecream import ic
import json


def make_all_histos(df,datatype="Recon",hists_2d=False,hists_1d=False,hists_overlap=False,saveplots=False):

    var_prefix = ""
    if datatype=="Gen":
        var_prefix = "Gen"
    vals = df.columns.values

    output_dir = "pics/"

    
    with open('utils/histo_config.json') as fjson:
        hftm = json.load(fjson)
    config = hftm["Ranges"][0]


    #Create set of 2D histos from JSON Specifications
    if hists_2d:
        for item in hftm:
            #print("in loop")
            hm = hftm[item][0]
            if hm["type"] == "2D":

                x_data = df[var_prefix+hm["data_x"]]
                y_data = df[var_prefix+hm["data_y"]]
                var_names = [hm["label_x"],hm["label_y"]]
                config_xy = [config[hm["type_x"]],config[hm["type_y"]]]
                ranges =  [config_xy[0][0],config_xy[1][0]]
                units = [config_xy[0][1],config_xy[1][1]]
                output_dir = "plots/hists_2d/{}/".format(datatype)
                title = "{} vs. {}, {}".format(var_names[0],var_names[1],datatype)
                filename = hm["filename"] # not currently used!

                #"Generated Events"

                make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                                filename=filename,units=units)

    #Create set of 1D histos
    if hists_1d:
        for x_key in vals:
            print("Creating 1 D Histogram for: {} ".format(x_key))
            xvals = df[x_key]
            make_histos.plot_1dhist(xvals,[x_key,],ranges="none",second_x="none",
                    saveplot=saveplots,pics_dir=output_dir,plot_title=x_key)

    
    # x_data = df_small_gen["GenxB"]
    # y_data = df_small_gen["GenQ2"]
    # x_data = df_small_gen["GenxB"]
    # y_data = df_small_gen["GenW"]
    # y_data = df_small_gen["Gent"]
    # x_data = df_small_gen["Genphi1"]
    # y_data = df_small_gen["GenPtheta"]
    # x_data = df_small_gen["GenPphi"]

    #Create set of overlapping 1D histos
    #       FILL OUT THIS SECTION XXXXXXXXXXXXXX
  





