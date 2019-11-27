__author__ = 'Richard Diehl Martinez'
''' Main training loop
'''

import click
from dataset import TextDataset
import logging

@click.command()
@click.argument('data_path')
def main(data_path):
    logging.info("Getting data from: {path}".format(path=data_path))
    data = TextDataset(data_path, is_train=True)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=20)
    main()
