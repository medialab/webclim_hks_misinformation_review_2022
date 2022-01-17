import os

import pandas as pd
import matplotlib.pyplot as plt


def import_data(file_name, folder=None):

    if folder:
        data_path = os.path.join(".", "data", folder, file_name)
    else:
        data_path = os.path.join(".", "data", file_name)
    
    df = pd.read_csv(data_path, low_memory=False)

    return df


def export_data(df, file_name, folder=None):

    if folder:
        csv_path = os.path.join(".", "data", folder, file_name + '.csv')
    else:
        csv_path = os.path.join(".", "data", file_name + '.csv')

    df.to_csv(csv_path, index=False)
    print("The '{}' file has been printed in the '{}' folder.".format(
        csv_path.split('/')[-1], csv_path.split('/')[-2])
    )


def save_figure(figure_name):

    figure_path = os.path.join(".", "figure", figure_name + '.png')
    plt.savefig(figure_path)

    print(
        '\n' + figure_name.upper() + '\n',
        "The '{}' figure has been saved in the '{}' folder.\n"\
            .format(figure_path.split('/')[-1], figure_path.split('/')[-2])
        )