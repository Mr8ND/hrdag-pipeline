import unittest2 as unittest
import pandas as pd
from main import clustering_step, save_obj_pickle


class MainFunctionTests(unittest.TestCase):

	def test_clustering_step_pbexample_function(self):

		hashids_set = set(['A', 'B', 'C', 'D', 'E', 'F'])
		cp = pd.DataFrame.from_records([('A', 'B', 0.9), ('A', 'C', 0.4),
				('A', 'D', 0.6), ('A', 'E', 0.3), ('B', 'C', 0.6),
				('C', 'F', 0.1 ), ('E', 'F', 0.97),	('D', 'E', 0.95),
				('D', 'F', 0.65)], 
				columns=['hash_1', 'hash_2', 'xgb_prob'])
		cp.set_index(['hash_1', 'hash_2'], drop=False, inplace=True)

		save_obj_pickle(obj=hashids_set, directory='inputs/', filename_out='mock_hashids.pkl')
		self.assertEqual([['E', 'D', 'F'], ['A', 'C', 'B']], clustering_step('inputs/mock_hashids.pkl', None, cp=cp))


if __name__ == '__main__':
	unittest.main()