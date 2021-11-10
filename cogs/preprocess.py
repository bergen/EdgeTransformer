# Licensed under the Apache License, Version 2.0
# Adapted from Ontanon et al. (2021), Making Transformers Solve Compositional Tasks

import collections
import copy
import dataclasses
import json
from typing import List, Mapping, Sequence, Tuple

DATA_DIR = './data/'  # Directory containing original tsv data files.
OUTPUT_DIR = './data/google_cogs/'  # Directory to write sequence tagged json files.
#DATA_FILES = [f'{x}.tsv' for x in ('dev', 'gen', 'test', 'train', 'train_100')]
DATA_FILES = [f'{x}.tsv' for x in ['gen_dev']]

@dataclasses.dataclass
class RawExample:
  input: str
  target: str
  distribution: str

@dataclasses.dataclass
class LabeledExample:
  tokens: List[str]  # Raw input tokens.
  parent: List[int]  # Index of parent token or `-1` for no parent.
  role: List[str]  # Role of token.
  category: List[str]  # Only "VERB" for verbs and "CNOUN" for common nouns; other words get empty category.
  noun_type: List[str]  # Whether this is a definite or indefinite noun (or neither).
  verb_name: List[str]  # Name of verb if token is a verb.
  distribution: str

def tokenize_input(input: str) -> List[str]:
  return input.split(' ')

def split_clauses(target: str) -> List[str]:
  compact_target = ''.join(target.split(' '))
  targets = compact_target.split('AND')
  targets = targets[0].split(';') + targets[1:]
  return targets

def get_arguments(clause: str) -> List[str]:
  assert clause[-1] == ')'
  idx = clause.find('(')
  assert idx != -1
  args = clause[(idx + 1):-1].split(',')
  assert 1 <= len(args) <= 2
  return args

def get_predicate(clause: str) -> str:
  start_idx = 1 if clause.startswith('*') else 0
  end_idx = clause.find('(')
  assert end_idx != -1
  return clause[start_idx:end_idx]

def is_index_arg(arg: str) -> bool:
  return arg.startswith('x_')

def get_arg_index(arg: str) -> int:
  return int(arg[2:])

def is_noun_clause(clause: str) -> bool:
  return '.' not in clause

def get_preposition(clause: str) -> str:
  marker = '.nmod.'
  idx = clause.find(marker)
  assert idx != -1
  return clause.split('(')[0][idx + len(marker):]

def is_proper_noun(token: str) -> bool:
  return token[0].isupper() and token not in ('A', 'An', 'The', 'TV')

NONE_ROLE = ''

class LabeledExampleBuilder:

  def __init__(self, raw_example: RawExample):
    tokens = tokenize_input(raw_example.input)
    self._example = LabeledExample(
        tokens=list(tokens),
        parent=[-1] * len(tokens),
        role=[NONE_ROLE] * len(tokens),
        category=[NONE_ROLE] * len(tokens),
        noun_type=[NONE_ROLE] * len(tokens),
        verb_name=[NONE_ROLE] * len(tokens),
        distribution=raw_example.distribution)

    self._idx_by_token = collections.defaultdict(list)
    for i, token in enumerate(tokens):
      self._idx_by_token[token].append(i)

    if self._process_primitive_example(raw_example):
      return

    self._mark_proper_noun_category()
    clauses = split_clauses(raw_example.target)
    clauses = self._remove_redundant_xcomp_agent(clauses)
    clauses = self._process_nouns(clauses)
    clauses = self._process_prepositions(clauses)
    # All remaining clauses should be for verbs now.
    self._process_verbs(clauses)

  def build(self) -> LabeledExample:
    return copy.deepcopy(self._example)

  def _process_primitive_example(self, raw_example: RawExample) -> bool:
    """Special handling if `raw_example` is primitive; no-op otherwise.

    Primitive examples are just single-word inputs; either a single noun
    or single verb in particular.  They can have LAMBDA notation in the target
    that we'll basically ignore for sequence labeling.

    Here's an example of a primitive example:
      input: touch
      target: LAMBDA a . LAMBDA b . LAMBDA e . touch . agent ( e , b ) AND touch . theme ( e , a )
      distribution: primitive

    Returns True if `raw_example` is primitive; False otherwise.
    """
    if raw_example.distribution != 'primitive':
      return False

    assert len(self._example.tokens) == 1

    marker_str = 'LAMBDA'
    idx1 = raw_example.target.find(marker_str)
    if idx1 == -1:
      # Proper nouns don't have any "LAMBDA"
      self._example.category[0] = 'PNOUN'
      return True
    idx2 = raw_example.target.find(marker_str, idx1 + 1)

    if idx2 == -1:
      # Only 1 "LAMBDA", so the primitive is a common noun.
      self._example.category[0] = 'CNOUN'
    else:
      # More than 1 "LAMBDA", so the primitive is a verb.
      self._example.category[0] = 'VERB'
      verb = self._example.tokens[0]
      assert f' {verb} ' in raw_example.target
      self._example.verb_name[0] = verb

    return True

  def _mark_proper_noun_category(self) -> None:
    for i, token in enumerate(self._example.tokens):
      if is_proper_noun(token):
        self._example.category[i] = 'PNOUN'

  def _remove_redundant_xcomp_agent(self, clauses: List[str]) -> List[str]:
    """Removes redundant agent clauses for xcomp relations.

    For example, for input "Audrey wished to crawl .", the clauses are
    ["wish.agent(x_1,Audrey)", "wish.xcomp(x_1,x_3)", "crawl.agent(x_3,Audrey)"]

    We remove "crawl.agent(x_3,Audrey)" since it's already implied by the
    other two clauses.  This is necessary since "Audrey" would otherwise
    have both "wish" and "crawl" as parents.  We only want "wish" as the
    parent.
    """
    xcomp_clauses = [x for x in clauses if '.xcomp' in x]
    for clause in xcomp_clauses:
      xcomp_args = get_arguments(clause)
      assert len(xcomp_args) == 2
      agent_clause_prefix = clause.split(',')[0].replace('xcomp', 'agent')
      agent_clauses = [x for x in clauses if x.startswith(agent_clause_prefix)]
      assert len(agent_clauses) == 1
      agent_args = get_arguments(agent_clauses[0])
      assert len(agent_args) == 2

      # Construct the clause we wish to remove.
      infinitive_verb = self._example.tokens[get_arg_index(xcomp_args[-1])]
      redundant_clause = (
          f'{infinitive_verb}.agent({xcomp_args[-1]},{agent_args[-1]})')
      clauses.remove(redundant_clause)

      # Set the verb_name for the infinitive verb.
      infinitive_idx = get_arg_index(xcomp_args[-1])
      self._example.verb_name[infinitive_idx] = infinitive_verb
    return clauses

  def _process_nouns(self, clauses: List[str]) -> List[str]:
    """Processes all noun clauses and returns all remaining clauses.

    This only populates `noun_type` labels.
    """
    remaining_clauses = []
    for clause in clauses:
      if is_noun_clause(clause):
        args = get_arguments(clause)
        assert len(args) == 1
        assert is_index_arg(args[0])
        idx = get_arg_index(args[0])
        self._example.category[idx] = 'CNOUN'
        assert self._example.noun_type[idx] == NONE_ROLE
        if clause.startswith('*'):
          self._example.noun_type[idx] = 'DEF'
        else:
          self._example.noun_type[idx] = 'INDEF'
      else:
        remaining_clauses.append(clause)

    # Sanity check that all remaining clauses have more than one argument.
    for clause in remaining_clauses:
      assert ',' in clause

    return remaining_clauses

  def _process_prepositions(self, clauses: List[str]) -> List[str]:
    remaining_clauses = []
    for clause in clauses:
      if '.nmod.' in clause:
        args = get_arguments(clause)
        assert len(args) == 2
        parent_noun_idx = get_arg_index(args[0])
        pp_noun_idx = get_arg_index(args[1])  # Prepositional phrase noun.

        # Find index of preposition
        preposition = get_preposition(clause)
        candidate_indices = self._idx_by_token[preposition]
        indices = [x for x in candidate_indices
                   if parent_noun_idx < x < pp_noun_idx]
        assert len(indices) == 1
        prep_idx = indices[0]

        # Make the preposition the parent of the prepositional phrase noun
        # and the parent noun the parent of the preposition.
        assert self._token_is_unassigned(pp_noun_idx)
        self._assign_token(pp_noun_idx, 'PP_NOUN', prep_idx)
        assert self._token_is_unassigned(prep_idx)
        self._assign_token(prep_idx, 'PREP', parent_noun_idx)
      else:
        remaining_clauses.append(clause)
    return remaining_clauses

  def _process_verbs(self, clauses: List[str]) -> None:
    remaining_clauses = []
    for clause in clauses:
      args = get_arguments(clause)
      assert len(args) == 2
      verb_idx = get_arg_index(args[0])
      if is_index_arg(args[1]):
        child_idx = get_arg_index(args[1])
      else:  # Proper noun argument.
        indices = self._idx_by_token[args[1]]
        assert len(indices) == 1
        child_idx = indices[0]
      predicate = clause.split('(')[0]
      verb, role = predicate.split('.')

      assert self._token_is_unassigned(child_idx)
      self._assign_token(child_idx, role, verb_idx)

      curr_name = self._example.verb_name[verb_idx]
      assert curr_name == NONE_ROLE or curr_name == verb
      self._example.verb_name[verb_idx] = verb
      self._example.category[verb_idx] = 'VERB'

  def _token_is_unassigned(self, index: int) -> bool:
    return (self._example.role[index] == NONE_ROLE and
            self._example.parent[index] == -1)

  def _assign_token(self, index: int, role: str, parent: int) -> None:
    self._example.role[index] = role
    self._example.parent[index] = parent


# Utilities to confirm that we can reconstruct the target from sequence labels.

@dataclasses.dataclass
class WrappedArg:
  index: int
  name: str = ''  # Only present for proper nouns.

  def to_str(self):
    if self.name:
      return self.name
    return f'x _ {self.index}'

@dataclasses.dataclass
class ReconstructedClause:
  predicate: str
  arguments: List[WrappedArg]
  is_definite: bool = False

  def to_str(self):
    arg_str = ' , '.join(x.to_str() for x in self.arguments)
    result = f'{self.predicate} ( {arg_str} )'
    if self.is_definite:
      result = '* ' + result
    return result


def reconstruct_target(example: LabeledExample) -> str:
  # Create noun clauses first, handling definite nouns differently
  # since they're separated and joined with ';' instead of 'AND'.
  definite_clauses = []
  other_clauses = []
  for i, noun_type in enumerate(example.noun_type):
    predicate = example.tokens[i]
    if predicate == 'TV':
      predicate = 'tv'
    if noun_type == 'DEF':
      definite_clauses.append(
          ReconstructedClause(predicate, [WrappedArg(i)], True))
    elif noun_type == 'INDEF':
      other_clauses.append(
          ReconstructedClause(predicate, [WrappedArg(i)]))

  # Add all other clauses, which correspond to non-empty roles.
  for i, role in enumerate(example.role):
    if role == 'PREP':
      # We capture the whole prepositional phrase clause from the 'PP_NOUN'
      # so we skip 'PREP' tokens for now.
      continue
    elif role == 'PP_NOUN':
      assert example.noun_type[i] != NONE_ROLE  # Never a proper noun.
      prep_idx = example.parent[i]
      preposition = example.tokens[prep_idx]
      parent_noun_idx = example.parent[prep_idx]
      parent_noun = example.tokens[parent_noun_idx]
      if parent_noun == 'TV':
        parent_noun = 'tv'
      predicate = f'{parent_noun} . nmod . {preposition}'
      assert example.noun_type[parent_noun_idx] != NONE_ROLE  # Never a proper noun.
      other_clauses.append(
          ReconstructedClause(
              predicate, [WrappedArg(parent_noun_idx), WrappedArg(i)]))
    elif role != NONE_ROLE:
      # Reconstruct verb clause
      verb_idx = example.parent[i]
      verb = example.verb_name[verb_idx]
      predicate = f'{verb} . {role}'
      other_clauses.append(
          ReconstructedClause(
              predicate, [WrappedArg(verb_idx), wrap_arg(i, example)]))
    if role == 'xcomp':
      verb = example.verb_name[i]
      parent_idx = example.parent[i]
      agent_idx = -1
      for j in range(len(example.tokens)):
        if example.role[j] == 'agent' and example.parent[j] == parent_idx:
          agent_idx = j
      assert agent_idx != -1
      other_clauses.append(
          ReconstructedClause(
              f'{verb} . agent', [WrappedArg(i), wrap_arg(agent_idx, example)]))

  def sort_key(clause: ReconstructedClause) -> Tuple[int, int]:
    if len(clause.arguments) > 1:
      second_idx = clause.arguments[1].index
    else:
      second_idx = -1
    return (clause.arguments[0].index, second_idx)

  other_clauses.sort(key=sort_key)

  clause_strings = [x.to_str() for x in definite_clauses]
  clause_strings.append(' AND '.join([x.to_str() for x in other_clauses]))
  return ' ; '.join(clause_strings)


def wrap_arg(index: int, example: LabeledExample) -> WrappedArg:
  """Creates a `WrappedArg` for the index, handling proper nouns appropriately.
  """
  if example.tokens[index][0].isupper():
    name = example.tokens[index]
  else:
    name = ''
  return WrappedArg(index, name)


# Label and write out all examples.


for filename in DATA_FILES:
  raw_examples = []
  with open(f'{DATA_DIR}{filename}') as f:
    for line in f.readlines():
      fields = [x.strip() for x in line.split('\t')]
      raw_examples.append(RawExample(*fields))

  examples = []
  for raw_example in raw_examples:
    example = LabeledExampleBuilder(raw_example).build()
    if raw_example.distribution != 'primitive':
      assert reconstruct_target(example) == raw_example.target
    examples.append(example)

  out_filename = filename.split('.')[0] + '_seqtag.jsonl'
  out_path = f'{OUTPUT_DIR}{out_filename}'
  with open(out_path, mode='w') as f:
    f.write('\n'.join(json.dumps(x.__dict__) for x in examples))