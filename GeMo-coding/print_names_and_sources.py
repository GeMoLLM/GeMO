# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple tool to iterate through the dataset, printing the name and source.

Example usage:

  print_names_and_sources /path/to/dataset/code_contests_train*
"""

import io
import sys

import riegeli
import json
# import numpy as np
import random

# np.random.seed(0)

import contest_problem_pb2


def _all_problems(filenames):
  """Iterates through all ContestProblems in filenames."""
  for filename in filenames:
    reader = riegeli.RecordReader(io.FileIO(filename, mode='rb'),)
    for problem in reader.read_messages(contest_problem_pb2.ContestProblem):
      yield problem


def _print_names_and_sources(args):
  filenames = [args[0]]
  # filetype, fileid = args[1].split('-')
  fileid = args[1]
  """Prints the names and sources of all ContestProblems in filenames."""
  cnt = 0
  d = {}
  
  for problem in _all_problems(filenames):
    d['source'] = contest_problem_pb2.ContestProblem.Source.Name(problem.source),
    d['name'] = problem.name
    
    if problem.name not in ['1215_A. Yellow Cards', '1474_A. Puzzle From the Future', '294_A. Shaass and Oskols', '854_A. Fraction', '902_A. Visiting a Friend', '246_A. Buggy Sorting', '675_A. Infinite Sequence', '9_A. Die Roll', '1451_A. Subtract or Divide', '1475_A. Odd Divisor', '295_A. Greg and Array', '366_A. Dima and Guards', '1301_A. Three Strings', '1325_A. EhAb AnD gCd', "1344_A. Hilbert's Hotel", '273_A. Dima and Staircase']:
      continue

    d['description'] = problem.description

    d['n_solution'] = len(problem.solutions)
    d['n_incorrect_solution'] = len(problem.incorrect_solutions)
    
    indices_py3_correct = [i for i, s in enumerate(problem.solutions) if s.language == contest_problem_pb2.ContestProblem.Solution.PYTHON3]
    indices_py3_incorrect = [i for i, s in enumerate(problem.incorrect_solutions) if s.language == contest_problem_pb2.ContestProblem.Solution.PYTHON3]
    
    indices_cpp_correct = [i for i, s in enumerate(problem.solutions) if s.language == contest_problem_pb2.ContestProblem.Solution.CPP]
    indices_cpp_incorrect = [i for i, s in enumerate(problem.incorrect_solutions) if s.language == contest_problem_pb2.ContestProblem.Solution.CPP]
    
    indices_java_correct = [i for i, s in enumerate(problem.solutions) if s.language == contest_problem_pb2.ContestProblem.Solution.JAVA]
    indices_java_incorrect = [i for i, s in enumerate(problem.incorrect_solutions) if s.language == contest_problem_pb2.ContestProblem.Solution.JAVA]
    
    indices_py_correct = [i for i, s in enumerate(problem.solutions) if s.language == contest_problem_pb2.ContestProblem.Solution.PYTHON]
    indices_py_incorrect = [i for i, s in enumerate(problem.incorrect_solutions) if s.language == contest_problem_pb2.ContestProblem.Solution.PYTHON]
    
    indices_unknown_correct = [i for i, s in enumerate(problem.solutions) if s.language == contest_problem_pb2.ContestProblem.Solution.UNKNOWN_LANGUAGE]
    indices_unknown_incorrect = [i for i, s in enumerate(problem.incorrect_solutions) if s.language == contest_problem_pb2.ContestProblem.Solution.UNKNOWN_LANGUAGE]
    
    d['n_solution_py3'] = len(indices_py3_correct)
    d['n_incorrect_solution_py3'] = len(indices_py3_incorrect)
    
    d['n_solutions_language'] = [len(indices_unknown_correct), 
                                 len(indices_py_correct),
                                 len(indices_cpp_correct),
                                 len(indices_py3_correct),
                                 len(indices_java_correct)]
    
    d['n_incorrect_solutions_language'] = [len(indices_unknown_incorrect),
                                          len(indices_py_incorrect),
                                          len(indices_cpp_incorrect),
                                          len(indices_py3_incorrect),
                                          len(indices_java_incorrect)]

    if len(indices_py3_correct) < 25:
      continue
        
    d['difficulty'] = problem.difficulty
    d['cf_index'] = problem.cf_index
    d['cf_tags'] = str(problem.cf_tags)
    d['cf_rating'] = problem.cf_rating

    # indices_sel_correct = np.array(indices_py3_correct)[np.random.choice(len(indices_py3_correct), 25, replace=False)]
    indices_sel_correct = random.sample(indices_py3_correct, 25)

    for i in range(25):
      d[f'solutions-{i}-id'] = int(indices_sel_correct[i])
      d[f'solutions-{i}-solution'] = json.dumps(problem.solutions[indices_sel_correct[i]].solution)
      d[f'solutions-{i}-language'] = json.dumps(problem.solutions[indices_sel_correct[i]].language)

    json.dump(d, open(f'./ext_{fileid}_problem_info_{cnt}.json', 'w'), indent=2)
    cnt += 1

if __name__ == '__main__':
  _print_names_and_sources(sys.argv[1:])
