from enum import Enum
from numpy import random as rnd
import numpy as np

threshold_plasticty = 1.75
delta_plasticity = 0.01

class SynapseType(Enum):
    INHIBITORY = "inhibitory"
    EXCITATORY = "excitatory"
    CUSTOM = "custom"

def gene_w_mat(p,size_a,size_b,mean_weight):
    connect_mat= np.where(p>rnd.rand(size_a,size_b),1,0) # Connectivity matrix based on contact probability
    possible_weights= rnd.exponential(mean_weight,size=(size_a,size_b)) # Synaptic weights for all possible synapses
    weights = np.where(connect_mat,possible_weights,0) # We keep only the weights of existing synapses
    return weights

def default_inhibitory_behavior(activity_source,w):
    a = activity_source
    to_apply = lambda x : np.clip(x-(activity_source@w),0,np.inf)
    return to_apply #target pop activity

def default_excitatory_behavior(activity_source,w):
    a = activity_source
    to_apply =lambda x : x+(activity_source@w)
    return to_apply #target pop activity

"""Mechanism for parallel fibers plasticity. WARNING : the parallel fibers weight would be taken
into account for the amount of growth as it as a direct impact on the calcium transient
WARNING: many arbitrary choices are made to compute the plasticity in such an abstract model
experimenting with different rules maybe necessary"""
def parallel_fiber_plasticity(activity_source,calcium_target,parallel_fibers_w): #TODO spontaneous growth

    growth_matrix = np.tile(activity_source, (len(calcium_target), 1)) # matrix indicating how much each parallel fiber should grow
    growth_matrix += calcium_target # use broadcasting to add the calcium concentration 
    #induced by climbing fibers to all granule cells activity vector (each duplicated row)
    growth_matrix = np.where(parallel_fibers_w>0,growth_matrix,0) # if there is no synapse, growth factor is equal to 0
    growth_matrix = np.where(growth_matrix>threshold_plasticty,-growth_matrix,growth_matrix)
    growth_matrix = growth_matrix*delta_plasticity
    to_apply = lambda x : x+growth_matrix
    return to_apply # to add to parrallel fibers weights
    
def update_copy_state(cere_state_copy,cere_state):

    for neuron_pop in cere_state["neuron_pop"]:
        for local_variables in cere_state["neuron_pop"][neuron_pop]["local_variables"]:
            np.copyto(cere_state_copy["neuron_pop"][neuron_pop]["local_variables"][local_variables],cere_state["neuron_pop"][neuron_pop]["local_variables"][local_variables])
        
        for synapse_pop in cere_state["synapse_pop"]:
            for local_variables in cere_state["synapse_pop"][synapse_pop]["local_variables"]:
                np.copyto(cere_state_copy["synapse_pop"][synapse_pop]["local_variables"][local_variables],cere_state["synapse_pop"][synapse_pop]["local_variables"][local_variables])
