#   Copyright (c) 2021 DeepEvolution Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is used to generate data for meta language models

import sys
import argparse
import multiprocessing
import numpy
import random
import os
from xenoverse.metalang import metalang_generator
from robofm.dataio import write_unified_record

def dump_data(path, idxes, configs):
    for idx in idxes:
        if(configs["sample_type"]=='tasks'):
            configs["output"] = path
        else:
            os.makedirs(path, exist_ok=True)
            temporary = os.path.join(path, ".robofm_tmp_%05d.npy" % idx)
            configs["output"] = temporary
        if configs["sample_type"] == "tasks":
            metalang_generator(**configs)
            continue
        try:
            metalang_generator(**configs)
            sequence = numpy.load(temporary, allow_pickle=False)
            write_unified_record(
                os.path.join(path, "record-%06d" % idx),
                {"tokens": sequence},
                producer={"name": "metalang", "version": "v2"},
            )
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
    

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generating Meta Language Tasks or Sequences')
    parser.add_argument('--sample_type', type=str, choices=['tasks', 'sequences'], default='sequences', help='Generate tasks or sequences')
    parser.add_argument('--task_file', type=str, default=None, help='Specify task file to generate from if the sample_type is sequences. Default will generate task on the fly.')
    parser.add_argument('--vocab_size', type=int, default=32)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--n_gram', nargs='+', type=int, default=[2,3,4,5,6], help="A [List of] length n used for generating tasks")
    parser.add_argument('--lambda_weight', type=float, default=5.0, help="Lambda weight multiplied for softmax sampling in MetaLangV2")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sequence_length', type=int, default=4096)
    parser.add_argument('--output_path', type=str, default="./metalm_data")
    parser.add_argument('--samples', type=int, default=1000, help="samples in each file")
    parser.add_argument('--file_number', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=1)

    args = parser.parse_args()

    configs = vars(args)
    configs["output_type"] = 'npy'
    configs["version"] = 'v2'

    processes = []
    output_path = args.output_path
    n_workers = args.workers
    file_number = args.file_number

    del configs["workers"]
    del configs["file_number"]
    del configs["output_path"]
    print("output to", output_path)

    for worker_id in range(n_workers):
        n_b = (file_number * worker_id) // n_workers
        n_e = (file_number * (worker_id + 1)) // n_workers
        if n_b >= n_e:
            continue

        print("start processes generating %05d to %05d" % (n_b, n_e))
        process = multiprocessing.Process(target=dump_data, 
                args=(output_path, range(n_b, n_e), configs))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"worker {process.pid} failed with exit code {process.exitcode}")
