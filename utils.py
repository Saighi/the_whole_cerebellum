from enum import Enum
from numpy import random as rnd
import numpy as np

threshold_plasticty = 1.75
delta_plasticity = 0.001

class SynapseType(Enum):
    INHIBITORY = "inhibitory"
    EXCITATORY = "excitatory"
    CUSTOM = "custom"

# def gene_w_mat(p,size_a,size_b,mean_weight):
#     connectivity= np.where(p>rnd.rand(size_a,size_b),1,0) # Connectivity matrix based on contact probability
#     possible_weights= rnd.exponential(mean_weight,size=(size_a,size_b)) # Synaptic weights for all possible synapses
#     weights = np.where(connectivity,possible_weights,0) # We keep only the weights of existing synapses
#     return weights,connectivity

def gene_w_inclusive(nb_connection,size_a,size_b,mean_weight,minimum_connect):
    connectivity = np.zeros((size_a,size_b))
    #possible_weights=  rnd.exponential(mean_weight,size=(size_a,size_b)) # Maybe more efficient to create singular inhibition and activity patterns 
    possible_weights= np.full((size_a,size_b),mean_weight)
    for source in range(size_a):
        connected_ind=rnd.choice(size_b,nb_connection,replace=False)
        connectivity[source,connected_ind]=1

    for target in range(size_b):
        if np.sum(connectivity[:,target])<minimum_connect: #if one neuron of the target population is not connected to enough neurons of source population
            source_connected_ind=rnd.choice(np.arange(size_a),minimum_connect,replace=False)
            connectivity[source_connected_ind,target]=1

    weights = np.where(connectivity,possible_weights,0) # We keep only the weights of existing synapses
    return weights,connectivity


def gene_w_exclusive(nb_connection,size_a,size_b,mean_weight):
    target_pop_indexes= np.arange(size_b)
    connectivity = np.zeros((size_a,size_b))
    #possible_weights= rnd.exponential(mean_weight,size=(size_a,size_b)) # Maybe more efficient to create singular inhibition and activity patterns 
    possible_weights= np.full((size_a,size_b),mean_weight)
    for i in range(size_a):
        target_pop_indexes_ind = np.arange(len(target_pop_indexes))
        if len(target_pop_indexes)>nb_connection:
            connected_ind=rnd.choice(target_pop_indexes_ind,nb_connection,replace=False)
        else:
            connected_ind=rnd.choice(target_pop_indexes_ind,len(target_pop_indexes),replace=False)
        connectivity[i,target_pop_indexes[connected_ind]]=1
        target_pop_indexes=np.delete(target_pop_indexes,connected_ind)
    weights = np.where(connectivity,possible_weights,0) # We keep only the weights of existing synapses
    return weights,connectivity


def default_inhibitory_behavior(activity_source,w):
    a = activity_source
    to_apply = lambda x : x-(activity_source@w)
    return to_apply #target pop activity

def default_excitatory_behavior(activity_source,w):
    a = activity_source
    to_apply =lambda x : x+(activity_source@w)
    return to_apply #target pop activity

"""Mechanism for parallel fibers plasticity. WARNING : the parallel fibers weight would be taken
into account for the amount of growth as it as a direct impact on the calcium transient
WARNING: many arbitrary choices are made to compute the plasticity in such an abstract model
experimenting with different rules maybe necessary"""
def parallel_fiber_plasticity(activity_source,calcium_target,connectivity): #TODO spontaneous growth

    growth_matrix = np.tile(activity_source, (len(calcium_target), 1)).T# matrix indicating how much each parallel fiber should grow
    growth_matrix += (calcium_target)+0.1 # use broadcasting to add the calcium concentration and spontaneous growth
    #induced by climbing fibers to all granule cells activity vector (each duplicated row)
    growth_matrix = np.where(connectivity,growth_matrix,0) 
    growth_matrix = np.where(growth_matrix>threshold_plasticty,-growth_matrix,growth_matrix)
    growth_matrix = growth_matrix*delta_plasticity
    to_apply = lambda x : np.clip(x+growth_matrix,a_min=0,a_max=None) # WARNING: could be problemetic if many events modify the synaptic weight. 
    # If it is the case the clip should be outside of the event function. But here it is ok as this function is the only one modifying the
    return to_apply # to add to parrallel fibers weights

def parallel_fiber_plasticity_2(activity_source,calcium_target,connectivity): #TODO spontaneous growth

    growth_matrix = np.tile(activity_source, (len(calcium_target), 1)) # matrix indicating how much each parallel fiber should grow$
    growth_matrix += calcium_target # use broadcasting to add the calcium concentration 
    #induced by climbing fibers to all granule cells activity vector (each duplicated row)
    growth_matrix = np.where(connectivity,growth_matrix,0) 
    growth_matrix = np.where(growth_matrix>threshold_plasticty,threshold_plasticty-growth_matrix,growth_matrix)
    growth_matrix = growth_matrix*delta_plasticity
    to_apply = lambda x : np.clip(x+growth_matrix,a_min=0,a_max=None) # WARNING: could be problemetic if many events modify the synaptic weight. 
    # If it is the case the clip should be outside of the event function. But here it is ok as this function is the only one modifying the
    return to_apply # to add to parrallel fibers weights
    
def update_copy_state(cere_state_copy,cere_state):

    for neuron_pop in cere_state["neuron_pop"]:
        for local_variables in cere_state["neuron_pop"][neuron_pop]["local_variables"]:
            np.copyto(cere_state_copy["neuron_pop"][neuron_pop]["local_variables"][local_variables],cere_state["neuron_pop"][neuron_pop]["local_variables"][local_variables])
        
        for synapse_pop in cere_state["synapse_pop"]:
            for local_variables in cere_state["synapse_pop"][synapse_pop]["local_variables"]:
                    np.copyto(cere_state_copy["synapse_pop"][synapse_pop]["local_variables"][local_variables],cere_state["synapse_pop"][synapse_pop]["local_variables"][local_variables])
