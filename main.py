# -*- coding: utf-8 -*-

import pandas as pd
from features.string_features import *
from features.simple_features import *
from features.hrdag_features import *
from features.arabic_rules import *
from utils.parser_utils import nameCleanerFunction
import pickle


# Importing Data
def import_dataset(filename, encoding='utf-8', separator='|'):
	return pd.DataFrame.from_csv(filename, encoding=encoding, sep=separator)


# Checking Data
def df_column_cleaner(df, id_cols_input, name_cols_input, dod_cols_input, loc_cols_input):

	list_check = [(id_cols_input, ['id_1', 'id_2']),
					(name_cols_input, ['name_1', 'name_2']),
					(dod_cols_input, ['date_of_death_1', 'date_of_death_2']),
					(loc_cols_input, ['location_1', 'location_2'])]

	output_col_name = []

	for input_list, output_list in list_check:
		if input_list != output_list:
			for j, input_name in enumerate(input_list):
				df[output_list[j]] = df[input_name]
		output_col_name += output_list

	return df[output_col_name]


def data_cleaner(df, arabic_cols=['name_1', 'name_2', 'location_1', 'location_2'], na_removal=True):

	df_copy = df.copy()

	for col in arabic_cols:
		df_copy[col] = df[col].apply(lambda x: nameCleanerFunction(x))
	if na_removal:
		df_copy.dropna(axis=0, how='any')
	return df_copy


# Create Arabic Rule Versions of names

def apply_arabic_rules(df, arabic_rules_dict, col_names=['name_1', 'name_2']):

	for name_col in col_names:
		for suff, arabic_func in arabic_rules_dict.iteritems():
			df[name_col+suff] = df[name_col].apply(arabic_func)

	return df


# Apply features to data
def apply_features(df, simple_feat_dict, hrdag_feat_dict, string_feat_dict, arabic_rules_dict):

	#final_feature_list = []

	#hrdag features applied
	for func_name, tuple_obj in hrdag_feat_dict.iteritems():
		func, m1, m2 = tuple_obj
		df[func_name] = df[[m1,m2]].apply(lambda x: func(x[0], x[1]), 1)
		#final_feature_list.append(func_name)

	#both simple and strings features applied
	suff_vec = [''] + arabic_rules_dict.keys()
	full_features_dict = dict(simple_feat_dict.items() + string_feat_dict.items())

	for func_name, tuple_obj in full_features_dict.iteritems():
		func, m1, m2 = tuple_obj
		for suff in suff_vec:
			df[func_name + suff] = df[[m1+suff, m2+suff]].apply(lambda x: func(x[0], x[1]), 1)
			#final_feature_list.append(func_name + suff)

	#return df, final_feature_list
	return df


# Drop all not-necessary data
def select_features_for_classification(df):

	feature_list = pickle.load(open('data/col_names_class.pkl', 'rb'))
	df_selected = df[feature_list]
	
	if isinstance(df_selected, pd.DataFrame):
		df_selected = df_selected.as_matrix()

	return df_selected


# Run Classification and attach results (match or not match)
# XGboost object need to be imported here

def run_classification(df, xgboost_filename):
	pass


# HDF file format
# Create HDF format

def create_hdf_file(df, hash_cols = ['id_1', 'id_2'], thresh_col='threshold'):
	pass


# Patrick's code
def clustering_step(input_filename, hdf_filename):
	pass


if __name__ == '__main__':

	id_name_col = ['id_1', 'id_2']
	input_name_col = ['match_1','match_2']
	input_dod_col = ['date_of_death_1', 'date_of_death_2']
	input_loc_col = ['location_1', 'location_2']
	
	mock_df_filename = 'data/mock_dataset_hrdag_pipeline.csv'
	df = import_dataset(mock_df_filename)
	cleaned_df = df_column_cleaner(df, id_name_col, input_name_col, input_dod_col, input_loc_col)
	cleaner_df = data_cleaner(cleaned_df)

	full_df = apply_arabic_rules(df, arabic_rules_dict=arabic_rules_dict)
	features_df = apply_features(full_df, simple_feat_dict, hrdag_feat_dict, string_feat_dict, arabic_rules_dict)

	print select_features_for_classification(features_df)
