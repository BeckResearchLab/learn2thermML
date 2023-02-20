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
            logger.info(f'Splitting dataset randomly with test frac {frac}')
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


def get_balance_data_sizes(n_class_a: int, n_class_b: int, desired_balance: float=0.5, max_upsampling: float=0.0):
    """Determine training final class data sizes to meet a desired
    
    We have some difference in number of data between the majority class. 
    a is the number of minority class, b is the number of majority class, so b-a is the difference in class sizes.

    The original data balance is Bi=a/(a+b). To fix the balancing problem we can upsample u data points from the minority class,
    and downsample d examples from the majority. To achieve the desired balance Bf:

    Bf = a+u/(a+u+b-d)

    The desired maximum fraction of upsampleing should be specified:
    u/a <= Fu

    The remaining portion to achieve Bf is done by downsampling.

    Parameters
    ----------
    n_class_a : int
        Original size of class a
    n_class_b : int
        Original size of class b
    desired_balance : float, default 0.5
        Final data balance between classes, must be > 0.0 and <= 0.5
    max_upsampling : float, default 0.0
        Fraction of original minority class to supsample. 0.0 means none, 1.0 means final size 200% original size

    Returns
    -------
    int, int : final data sizes to class a and b
    """
    # check for valid input
    if desired_balance > 0.5 or desired_balance <= 0.0:
        raise ValueError("desired_balance must be between 0.0 and 0.5")

    # determinine the majority
    if n_class_b > n_class_a:
        maj_class = 'b'
        maj_size = n_class_b
        min_size = n_class_a
    elif n_class_b < n_class_a:
        maj_class = 'a'
        maj_size = n_class_a
        min_size = n_class_b
    else:
        logger.info("Dataset already perfectly balanced, making no change.")
        return n_class_a, n_class_b

    # check that the data has not already met the criteria 
    if min_size/(min_size + maj_size) >= desired_balance:
        logger.info(f"Asked for minimum balance of {desired_balance}, but the data is already  more balanced: {min_size/(min_size + maj_size)}, making no change.")
        return n_class_a, n_class_b

    # Compute amount to upsample
    # determine if the max upsampling will put us over the desired balance
    u_tmp = int(max_upsampling * min_size) # assuming we do the maximum amount of upsampling
    if ((min_size + u_tmp) / (min_size + u_tmp + maj_size)) > desired_balance:
        u = int((desired_balance * (min_size + maj_size) - min_size)/(1-desired_balance))
        d = 0
    else:
        u = u_tmp
        d = int(u + min_size + maj_size - (min_size + u)/desired_balance)

    # return final sizes
    logger.info(f"Suggesting upsampling minority class by {u} and downsampling majority class by {d} to meet balance of {desired_balance}")
    min_size_final = min_size + u
    maj_size_final = maj_size - d
    if maj_class == 'b':
        return min_size_final, maj_size_final
    else:
        return maj_size_final, min_size_final

