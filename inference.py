#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas import DataFrame
from joblib import load
BEST_MODEL_PATH = "resources/best_model.joblib" #change this line as you wish

model = load(BEST_MODEL_PATH)

def inference(path: str)->List[int]:
    '''
    path: a DataFrame
    result is the output of function which should be 
    somethe like: [0,1,1,1,0]
    0 -> Lost
    1 -> Won
    '''
    result = []
    ## your code starts here
    #model.predict()
    
    ## your code ends here
    return result
    

