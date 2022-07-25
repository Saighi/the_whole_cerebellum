import numpy as np

class NeuronPop:
    """ Implement a very simplified population of neurons as an activity vector.
    
    Parameters
    ----------
    size: int
        the number of neurons
    activity : array (Default: None)
    local_variables: dict() 
        Store any variables that could be useful and not captured by the activity vector.
        Examples: Calcium concentration, activity from one synaptic source etc.
    local_constant: dict()
        Same thing for constant values.
    """
    def __init__(
        self,
        name,
        size,
        activity = None,
        min_activity = 0,
        max_activity = 1
    ):
        self.size = size

        if activity:
            self.activity = activity
        else :
            self.activity = [0 for i in range(size)]

        self.local_variables = dict() 
        self.local_constants = dict() 
        self.min_activity = min_activity
        self.max_activity = max_activity
        self.name = name
    
    def __str__(self) -> str:
        return self.name
    
    def add_local_variable(self,name,value):
        self.local_variables[name]=value
    
    def add_local_constant(self,name,value):
        self.local_constants[name]=value
    
    def change_local_variable(self,name,value):
        self.local_variables[name]=value

    def read_local_variable(self,name):
        return self.local_variables[name]
    
    def read_local_constant(self,name):
        return self.local_constants[name]
        
    