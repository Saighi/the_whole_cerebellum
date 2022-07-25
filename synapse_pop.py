import numpy as np
from enum import Enum
from utils import *
from neuron_pop import NeuronPop

class SynapsePop:
    """ Implement a very simplified population of synapses between two population of neurons.
    
    Parameters
    ----------
    source_pop: class instance
        The population of neurons from which come the synapses
    target_pop: class instance
        The targeted population of neurons
    contact_prob: double
        The probability of making synapse onto the target population.
    mean_weight: double
        Mean value of a synaptic weight 
    type: enum
        Can be an inhibitory or excitatory synapse (implicitly define the source population type)
    local_variables: dict() 
        Store any variables that could be useful and not already captured.
    local_constant: dict()
        Same thing for constant values.
    """

    def __init__(
        self,
        name,
        source_pop,
        target_pop,
        contact_prob,
        mean_weight,
        type,
    ):
        self.source_pop = source_pop
        self.target_pop = target_pop
        self.contact_prob = contact_prob
        self.weights= gene_w_mat(contact_prob,source_pop.size,target_pop.size,mean_weight)
        self.type = type
        self.local_variables = dict() 
        self.local_constant = dict() 
        self.name = name

        if type == SynapseType.INHIBITORY:
            self.apply_behavior = default_inhibitory_behavior
        elif type == SynapseType.EXCITATORY:
            self.apply_behavior = default_excitatory_behavior # Default behavior of a synapse is to add a fraction of the source activity the target activity 
        elif type == SynapseType.CUSTOM:
            print("WARNING: Synaptic group "+str(self)+" needs a custom behavior to be defined.")
        
    def __str__(self) -> str:
        return self.name

    def define_behavior (self,new_behavior):
        self.apply_behavior = new_behavior

    