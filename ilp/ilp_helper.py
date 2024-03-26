import os
import re
import subprocess
from sys import platform

import numpy as np
import pandas as pd

from helper import OS_IS

PREFIX = "coinrun_ppo_actor_checkpoint"

SETUP_DICT = {'wsl': True,
              'clingo': 'clingo',
              'ILASP': '~/ILASP',
              'FastLAS': 'FastLAS',
              'ltlfilt': 'ltlfilt'
              }
if platform == "linux" or platform == "linux2":
    SETUP_DICT["wsl"] = False


def clean_path(filename):
    if platform == "linux" or platform == "linux2":
        return re.sub("%2E", ".", filename)
    return filename


def new_file(folder, name, suffix=""):
    files = os.listdir(folder)
    nums = [re.search(fr"{name}(\d*)\.{suffix}", x).group(1) for x in files if re.search(name, x)]
    if nums == []:
        num = "0000"
        return os.path.join(folder, f"{name}{num}.{suffix}")
    length = max([len(n) for n in nums])
    last = max([int(n) for n in nums])
    next = str(last + 1)
    num = ''.join(['0' for _ in range(length - len(next))] + [next])
    return os.path.join(folder, f"{name}{num}.{suffix}")


def create_cmd(param):
    cmd = create_cmd_list(param)
    if OS_IS == "Linux":
        return ' '.join(cmd)
    return cmd


def create_cmd_list(param):
    cmd = []
    if SETUP_DICT['wsl']:
        cmd.append('wsl')
    cmd.append(SETUP_DICT[param[0]])
    if len(param) == 1:
        return cmd
    cmd += param[1:]
    return cmd


def run_subprocess(cmd, newline, suppress=False, timeout=-1):
    # timed = timeout > 0

    if suppress:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True,
                             stderr=subprocess.DEVNULL)  # , start_new_session=timed)
    else:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)  # , start_new_session=timed)
    # if timed:
    #     try:
    #         p.wait(timeout=timeout)
    #     except subprocess.TimeoutExpired:
    #         print("Subprocess Timeout")
    #         os.kill(p.pid, signal.CTRL_C_EVENT)
    #         subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)])
    #         return "Timeout"
    output = p.communicate()[0]

    output = '\n'.join(output.decode("utf-8").split(newline))
    return output


def append_to_csv_if_exists(df, filename):
    if os.path.isfile(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        df.to_csv(filename, mode="w", header=True, index=False)


def write_string_to_file(output, filename):
    file = open(filename, 'w', newline='\n')
    file.write(output)
    file.close()


def extract_clingo_solution(output):
    lines = output.split("\n")
    answer = [lines[i - 1] for i, line in enumerate(lines) if line == "SATISFIABLE"]
    return answer[0].split(" ")


def read_file(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    return lines


def get_latest_file_matching(pattern, n, folder=""):
    if folder == "":
        files = os.listdir()
    else:
        files = [os.path.join(folder, x) for x in os.listdir(folder)]
    sl_files = {x: os.path.getmtime(x) for x in files if re.search(pattern, x)}
    if n == 1:
        return max(sl_files, key=sl_files.get)
    sl_files = dict(sorted(sl_files.items(), key=lambda item: item[1]))
    return list(sl_files.keys())[-n]


def get_specs_name(folder, unique_id=""):
    files = os.listdir(folder)
    actor_file = [re.sub(r"\.index", "", file) for file in files if
                  re.search(r"coinrun_ppo_actor_checkpoint", file) and re.search(r"\.index", file)][0]
    try:
        slurm_file = [file for file in files if re.search("slurm", file)][0]
        text = read_file(os.path.join(folder, slurm_file))
        title = [re.sub(r"\s", "", text[i + 1]) for i, x in enumerate(text) if re.search("Starting For:", x)][0]
    except IndexError:
        title = get_deets_from_id(unique_id)
    specs = extract_dict(title)
    specs = dict_str_to_val(specs)
    try:
        del specs["n_levels"]
    except Exception:
        pass
    return specs, os.path.join(folder, actor_file)


def dict_str_to_val(specs):
    specs = {k: (bool(v == "True") if v == "True" or v == "False" else (float(v) if re.search(r"\.", v) else int(v)))
             for k, v in
             specs.items()}
    return specs


def get_all_deets_from_id(unique_id):
    return get_deets_from_id(unique_id, pattern=r"trained_models/(.*)")


def get_deets_from_id(unique_id, pattern=r"trained_models/(.*)_n.{0,2}_levels"):
    df = pd.read_csv("trained_models/record.csv", index_col=0)
    y = [y for x, y in zip(df.filename, df.details) if re.search(unique_id, x)][0]
    title = re.search(pattern, y).group(1)
    return title


def extract_dict(title):
    text = re.split("_", title)
    text = text[[i + 1 for i, x in enumerate(text) if x == "checkpoint"][0]:len(text)]
    stn = ""
    deets = {}
    for ch in text:
        if re.search(r"\d+|True|False|\d+\.\d+", ch):
            deets[stn[:-1]] = ch
            stn = ""
        else:
            stn += f"{ch}_"
    return deets


def rgb_to_jpeg(arr, filename="your_file.jpeg"):
    from PIL import Image
    im = Image.fromarray(arr)
    im.save(filename)


def batch_to_state(arr):
    state = np.array(arr) / 255.0 - 0.5
    # if state.shape != (64, 64, 3):
    #     print(state.shape)
    state = state.astype("float32")
    return state


class RandomActor:
    def __init__(self, output_shape, seed):
        self.rng = np.random.default_rng(seed)
        self.output_shape = output_shape

    def __call__(self, observation):
        return self.rng.random(self.output_shape), 0


class NaiveActor:
    def __init__(self, output_shape, seed):
        self.rng = np.random.default_rng(seed)
        self.output_shape = output_shape

    def __call__(self, observation):
        # Naive coinrun policy; equal mix of UP (5) /RIGHT (7) and UP+RIGHT (8)
        x = self.rng.choice([5, 7, 8], self.output_shape[0])
        return np.eye(self.output_shape[1])[x], 0


def np_fill_nan(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    arr[mask] = arr[idx[mask]]


def rolling_mean(a, n=4):
    n -= 1
    a = np.array(a)
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[:n] = np.nan
    return ret / (n + 1)


def is_duplicated(input_list):
    seen = set()
    duplicated = []
    for x in input_list:
        if x in seen:
            duplicated.append(True)
        else:
            duplicated.append(False)
            seen.add(x)
    return duplicated


def dict_to_html_table(name_deets):
    specs = {"metric": list(name_deets.keys()), "value": list(name_deets.values())}
    df2 = pd.DataFrame(specs)
    text = df2.to_html(index=False)
    return text


def get_full_deets(unique_id):
    title = get_all_deets_from_id(unique_id)
    deets = extract_dict(title)
    return dict_str_to_val(deets)


def match(a, b):
    a = a.tolist()
    b = b.tolist()
    return np.array([b.index(x) for x in a if x in b])


def copy_whole_folder(lib_dir, to_dir):
    from distutils.dir_util import copy_tree
    copy_tree(lib_dir, to_dir)
