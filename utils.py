import logging
import os
import pandas as pd


def list_files(directory, report_suffix,  output_filename):
    for filename in os.listdir(directory):
        if report_suffix in filename:
            print(filename)


def create_output(directory, report_suffix,  output_filename):
    df_all = pd.DataFrame()

    for filename in os.listdir(directory):
        if report_suffix in filename:
            year = filename[0:4]
            df = pd.read_csv('/'.join([directory, filename]), sep=',', header=7, quotechar='"')
            logging.info(f'Reading {filename}.')

            df['year'] = year
            df_all = pd.concat([df_all, df],axis=0, ignore_index=True)
            logging.info(f'Dataframe for {year} successfully appended.')

        else:
            continue

    df_all.to_csv('/'.join([directory, output_filename]))
