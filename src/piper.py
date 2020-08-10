from sklearn.pipeline import Pipeline
from clean_helpers import *


def compile_pipe(transformers=[],transformer_names=[]):
    pipe_components = list(zip(transformer_names, transformers))
    return Pipeline(pipe_components)



def get_pipe(model):
  # add model to transformers and names
  # compile and return pipe
  transformers = [Preprocesser(), CustomEncoder(), CustomScaler()]
  names = ['Preprocesser', 'Encoder', 'Scaler', 'Model']
  tranformers.append(model)
  return compile_pipe(transformers=transformers,transformer_names=names)