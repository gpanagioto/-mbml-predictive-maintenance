# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
from datetime import date


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


        # load data from csv
    telemetry = pd.read_csv(input_filepath+'/'+'telemetry.csv')

    # format datetime field which comes in as string
    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")

    # load data from csv
    error_count = pd.read_csv(input_filepath+'/'+'error_count.csv')

    # format datetime field which comes in as string
    error_count['datetime'] = pd.to_datetime(error_count['datetime'], format="%Y-%m-%d %H:%M:%S")

    # load data from csv
    comp_rep = pd.read_csv(input_filepath+'/'+'comp_rep.csv')

    # format datetime field which comes in as string
    comp_rep['datetime'] = pd.to_datetime(comp_rep['datetime'], format="%Y-%m-%d %H:%M:%S")

    def lifespan(comp_rep0):
        comp_rep=comp_rep0.copy()
        points = comp_rep['machineID'].unique()
        final=pd.DataFrame()

        for i in points:
            df = comp_rep[(comp_rep['machineID']==i)][['datetime','machineID','comp1','comp2','comp3','comp4']]
            for comp in ['comp1','comp2','comp3','comp4']:
                life=comp+'_life'
                df[life] = df.apply(lambda row: row['datetime'] if row[comp]==0 else np.nan, axis=1)
                df[df[life].isna()==False].index
                df[life].fillna(method='backfill', inplace=True)
                df[life] = pd.to_datetime(df[life]) - df['datetime']
                df[life] = df[life].apply(lambda row: row.total_seconds()/86400)
            final=pd.concat([final,df],axis=0)
        return final.copy()

    comp_rep=lifespan(comp_rep)

    #concat df dataframe to final dataframe
    # load data from csv
    machines = pd.read_csv(input_filepath+'/'+'machines.csv')

    # turn "model" variable into dummy variables
    machines['model'] = machines['model'].astype('category')
    machines = pd.get_dummies(machines)


    # load data from csv
    failures = pd.read_csv(input_filepath+'/'+'failures.csv')

    # format datetime field which comes in as string
    failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
    #failures['failure'] = failures['failure'].astype('category')

    features = telemetry.merge(error_count, on=['datetime', 'machineID'], how='left')
    features = features.merge(comp_rep, on=['datetime', 'machineID'], how='left')
    features = features.merge(machines, on=['machineID'], how='left')
    # main target variable: "failure"
    labeled_features = features.merge(failures, on=['datetime', 'machineID'], how='left')

    labeled_features['comp1_fail'] = (labeled_features['failure'] == 'comp1').astype(int).replace(0, np.nan)
    labeled_features['comp2_fail'] = (labeled_features['failure'] == 'comp2').astype(int).replace(0, np.nan)
    labeled_features['comp3_fail'] = (labeled_features['failure'] == 'comp3').astype(int).replace(0, np.nan)
    labeled_features['comp4_fail'] = (labeled_features['failure'] == 'comp4').astype(int).replace(0, np.nan)

    labeled_features = labeled_features.fillna(method='bfill', limit=7) # fill backward up to 24h

    labeled_features['comp1_fail'] = (labeled_features['failure'] == 'comp1').astype(int)
    labeled_features['comp2_fail'] = (labeled_features['failure'] == 'comp2').astype(int)
    labeled_features['comp3_fail'] = (labeled_features['failure'] == 'comp3').astype(int)
    labeled_features['comp4_fail'] = (labeled_features['failure'] == 'comp4').astype(int)

    labeled_features.to_csv(output_filepath+'/'+'data_processed.csv')




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
