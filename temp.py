import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--wandb_tags', type=str, nargs='+')

    args = parser.parse_args()

    print(args.wandb_tags)
