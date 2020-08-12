from sklearn.pipeline import Pipeline
from clean_helpers import *


def compile_pipe(transformers=[],transformer_names=[]):
    pipe_components = list(zip(transformer_names, transformers))
    return Pipeline(pipe_components)



def get_pipe(model, continuous_cols, categorical_cols):
    
    transformers = [
                    Preprocesser(), 
                    CustomEncoder(categorical_cols=categorical_cols), 
                    CustomScaler(continuous_cols=continuous_cols), 
                    model
                    ]
    names = ['Preprocesser', 'Encoder', 'Scaler', 'Model']
    return compile_pipe(transformers=transformers,transformer_names=names)