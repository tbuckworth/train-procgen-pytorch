import argparse
import os
import numpy as np

from helper_local import run_subprocess, get_latest_file_matching

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1, help="Nth most recent batch to cancel")
    args = parser.parse_args()

    filename = get_latest_file_matching("hosts_.*.npy", args.n, "data")

    hosts = np.load(filename, allow_pickle='TRUE').item()

    for host in hosts.keys():
        for session_name in hosts[host]:
            cmd1 = f'ssh {host} "tmux send -t {session_name}.0 ^C^C ENTER exit ENTER"'
            run_subprocess(cmd1, "\\n", suppress=False)
