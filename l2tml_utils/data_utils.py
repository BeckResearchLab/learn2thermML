"""Commonly used utilities for the ML pipeline."""

from datasets import Dataset, DatasetDict
import sklearn.model_selection

from typing import Union

import logging
logger = logging.getLogger(__name__)

class DataSplitter:
    """Split dataset into train, val test, with options for splitting by taxonomy.
    
    Options for splitting:
    - randomly
    - keep taxa proteins together
    - keep proteins from a taxonomy level together

    Parameters
    """
    def __init__(self, dataset):
        if not isinstance(dataset, Dataset):
            raise ValueError('Expected Hugging face Dataset')
        else:
            self.dataset=dataset
        return

    @staticmethod
    def _get_taxonomy_at_level(taxonomy_str: str, level: int):
        splits = taxonomy_str.split()
        label = ' '.join(splits[:level+1])
        return label

    def split(
        self,
        splittype: Union[str, int],
        frac: float=0.15,
        splitting_col: str=None
    ):
        if splittype == 'random':
            logger.info('Splitting dataset randomly with test frac {frac}')
            return self.dataset.train_test_split(test_size=frac)
        elif type(splittype) == int:
            raise NotImplemented(
                'Splitting by taxa level not implemented')
            # assume defualt column specifying taxonomy if not given
            if splitting_col is None:
                splitting_col = 'taxonomy'
            
            # get rid of nan taxonomy
            self.dataset=self.dataset.filter(lambda e: e[splitting_col]!=None)

            # create function to get taxa at a level as labels (new column)
            def apply_add_taxlvl_column(example):
                example['tax_at_level'] = self._get_taxonomy_at_level(example[splitting_col], splittype)
                return example

            # apply the method
            self.dataset = self.dataset.map(apply_add_taxlvl_column)
            # we are splitting with groups specified by this new column
            splitting_col = 'tax_at_level'
        elif splittype == 'taxa_id':
            logger.info('Splitting dataset by taxa with test frac {frac}')
            if splitting_col is None:
                splitting_col = 'taxa_index'
        else:
            raise ValueError(
                f"Cannot interpret `splittype={splittype}`, see docs")
        # use the labels assigned to each example to split data keeping
        # data of the same label together
        group_shuffle_splitter = sklearn.model_selection.GroupShuffleSplit(
            n_splits=1, train_size=1-frac)
        for train_indexes, test_indexes in group_shuffle_splitter.split(X=None, groups=self.dataset[splitting_col]):
            pass
        train = self.dataset.select(train_indexes)
        test = self.dataset.select(test_indexes)

        return DatasetDict({'train':train, 'test':test})


def get_balance_data_sizes(n_class_a: int, n_class_b: int, desired_balance: float=0.5, upsample_frac: float=0.0):
    """Determine training final class data sizes to meet a desired
    
    We have some difference in number of data between the majority class. 
    a is the number of minority class, b is the number of majority class, so b-a is the difference in class sizes.

    The original data balance is Bi=a/(a+b). To fix the balancing problem we can upsample u data points from the minority class,
    and downsample d examples from the majority. To achieve the desired balance Bf:

    Bf = a+u/(a+u+b-d)

    The desired fraction of upsampleing should be specified:
    Fu = u/(u+d)

    Thus the final amounts can be specified:



    """
    return None