# -*- coding: utf-8 -*-

import pandas as pd
from features.string_features import *
from features.simple_features import *
from features.hrdag_features import *
from features.arabic_rules import *

# Importing Data

def import_dataset(filename, encoding='utf-8', separator='|'):
	return pd.DataFrame.from_csv(filename, encoding=encoding, sep=separator)


def df_column_cleaner(df, name_cols_input, dod_cols_input, loc_cols_input, match_col_input):
	list_check = [(name_cols_input, ['name_1', 'name_2']),
					(dod_cols_input, ['date_of_death_1', 'date_of_death_2']),
					(loc_cols_input, ['location_1', 'location_2']),
					(match_col_input, ['match_status'])]

	output_col_name = []

	for input_list, output_list in list_check:
		if input_list != output_list:
			for j, input_name in enumerate(input_list):
				df[output_list[j]] = df[input_name]
		output_col_name += output_list

	return df[output_col_name]




# Checking Data

# Create Arabic Rule Versions of names

# Apply features to data

# Drop all not-necessary data 

# Run Classification and attach results (match or not match)


# HDF file format

# Patrick's code


if __name__ == '__main__':

	input_name_col = ['match_1','match_2']
	input_dod_col = ['date_of_death_1', 'date_of_death_2']
	input_loc_col = ['location_1', 'location_2']
	input_matchstatus_col = ['match_status']
	
	mock_df_filename = 'data/mock_dataset_hrdag_pipeline.csv'
	df = import_dataset(mock_df_filename)
	print df_column_cleaner(df, input_name_col, input_dod_col, input_loc_col, input_matchstatus_col)