import argparse

from helper_local import add_pets_args
from pets.pets import run_pets



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_pets_args(parser)
    args = parser.parse_args()
    run_pets(args)
