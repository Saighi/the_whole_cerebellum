import numpy as np
from enum import Enum
from numpy import random as rnd

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

def gene_connectivity_inclusive(nb_connection, size_a, size_b, mean_weight, minimum_connect):
    connectivity = np.zeros((size_a, size_b))
    possible_weights = rnd.exponential(mean_weight, size=(
        size_a, size_b))  # Maybe more efficient to create singular inhibition and activity patterns
    # possible_weights= np.full((size_a,size_b),mean_weight)
    for source in range(size_a):
        connected_ind = rnd.choice(size_b, nb_connection, replace=False)
        connectivity[source, connected_ind] = 1

    for target in range(size_b):
        if np.sum(connectivity[:,
                  target]) < minimum_connect:  # if one neuron of the target population is not connected to enough neurons of source population
            source_connected_ind = rnd.choice(np.arange(size_a), minimum_connect, replace=False)
            connectivity[source_connected_ind, target] = 1

    weights = np.where(connectivity, possible_weights, 0)  # We keep only the weights of existing synapses
    return weights, connectivity


def generate_outside_microzone_connectivity(cere_info, cere_state, microzone_synapses):
    for key in cere_info["synapse_pop"]:
        nb_connect = cere_info["synapse_pop"][key]["nb_connect"]
        input_pop = cere_info["synapse_pop"][key]["i_o"][0]
        input_pop_size = cere_info["neuron_pop"][input_pop]["size"]
        target_pop = cere_info["synapse_pop"][key]["i_o"][1]
        target_pop_size = cere_info["neuron_pop"][target_pop]["size"]
        mean_weight = cere_info["synapse_pop"][key]["mean_weight"]
        minimum_connect = cere_info["synapse_pop"][key]["minimum_connect"]
        if key not in microzone_synapses:
            weight_matrice, connectivity = gene_connectivity_inclusive(nb_connect, input_pop_size, target_pop_size,
                                                                       mean_weight, minimum_connect)
            cere_state["synapse_pop"][key]["local_variables"]["weight"] = weight_matrice
            cere_state["synapse_pop"][key]["local_variables"]["connectivity"] = connectivity
    return cere_state


def dense_microzone_connectivity(microzone_size, size_source, size_target, mean_weight, pathway_source=None,
                                 pathway_target=None):
    connectivity = np.zeros((size_source, size_target))
    # possible_weights= rnd.exponential(mean_weight,size=(size_a,size_b)) # Maybe more efficient to create singular inhibition and activity patterns
    possible_weights = np.full((size_source, size_target), mean_weight)
    nb_microzone = int(1 / microzone_size)  # if microzone size is 0.01 we get 100 microzones

    if pathway_source is None:  # If sources neuron doesn't already belong to any pathway (in our case PC is the start of microzone pathway)
        randomly_chosen_sources = rnd.choice(size_source, size_source, replace=False)
        source_microzones = np.split(randomly_chosen_sources, nb_microzone)
        pathway_source = np.zeros(size_source, dtype=int)
        for idm, m in enumerate(source_microzones):
            for neuron_id in m:
                pathway_source[neuron_id] = idm
    else:
        source_microzones = [[] for _ in range(nb_microzone)]
        for idp, p in enumerate(pathway_source):
            source_microzones[p].append(idp)

    if pathway_target is None:
        randomly_chosen_target = rnd.choice(size_target, size_target, replace=False)
        target_microzones = np.split(randomly_chosen_target, nb_microzone)
        pathway_target = np.zeros(size_target, dtype=int)
        for idm, m in enumerate(target_microzones):
            for neuron_id in m:
                pathway_target[neuron_id] = idm
    else:
        target_microzones = [[] for _ in range(nb_microzone)]
        for idp, p in enumerate(pathway_target):
            target_microzones[p].append(idp)

    for m in range(nb_microzone):
        connectivity[source_microzones[m], target_microzones[m]] = 1

    weights = np.where(connectivity, possible_weights, 0)  # We keep only the weights of existing synapses

    return weights, connectivity, pathway_source, pathway_target


def generate_microzone_connectivity(cere_info, cere_state):
    """The synaptic connectivity between microzones follow a specific pattern that allows for each
    Purkinje cells to act on his own climbing fibers"""

    microzone_size = cere_info["microzone_size"]

    PC_size = cere_info["neuron_pop"]["PC"]["size"]
    DCN_size = cere_info["neuron_pop"]["DCN"]["size"]
    IO_size = cere_info["neuron_pop"]["IO"]["size"]

    mean_weight_PC_DCN = cere_info["synapse_pop"]["PC_DCN"]["mean_weight"]
    mean_weight_DCN_IO = cere_info["synapse_pop"]["DCN_IO"]["mean_weight"]
    mean_weight_IO_PC = cere_info["synapse_pop"]["IO_PC"]["mean_weight"]

    synapses_pop_states = dict()

    weights_PC_DCN, connectivity_PC_DCN, PC_pathway, DCN_pathway = dense_microzone_connectivity(microzone_size, PC_size,
                                                                                                DCN_size,
                                                                                                mean_weight_PC_DCN)
    cere_state["synapse_pop"]["PC_DCN"]["local_variables"]["weight"] = weights_PC_DCN
    cere_state["synapse_pop"]["PC_DCN"]["local_variables"]["connectivity"] = connectivity_PC_DCN

    weights_DCN_IO, connectivity_DCN_IO, _, IO_pathway = dense_microzone_connectivity(microzone_size, DCN_size, IO_size,
                                                                                      mean_weight_DCN_IO,
                                                                                      pathway_source=DCN_pathway)
    cere_state["synapse_pop"]["DCN_IO"]["local_variables"]["weight"] = weights_DCN_IO
    cere_state["synapse_pop"]["DCN_IO"]["local_variables"]["connectivity"] = connectivity_DCN_IO

    weights_IO_PC, connectivity_IO_PC, _, _ = dense_microzone_connectivity(microzone_size, IO_size, PC_size,
                                                                           mean_weight_IO_PC, pathway_source=IO_pathway,
                                                                           pathway_target=PC_pathway)
    cere_state["synapse_pop"]["IO_PC"]["local_variables"]["weight"] = weights_IO_PC
    cere_state["synapse_pop"]["IO_PC"]["local_variables"]["connectivity"] = connectivity_IO_PC

    return cere_state


def default_inhibitory_behavior(activity_source, w):
    a = activity_source
    to_apply = lambda x: x - (activity_source @ w)
    return to_apply  # target pop activity


def default_excitatory_behavior(activity_source, w):
    a = activity_source
    to_apply = lambda x: x + (activity_source @ w)
    return to_apply  # target pop activity


"""Mechanism for parallel fibers plasticity. WARNING : the parallel fibers weight would be taken
into account for the amount of growth as it as a direct impact on the calcium transient
WARNING: many arbitrary choices are made to compute the plasticity in such an abstract model
experimenting with different rules maybe necessary"""


def parallel_fiber_plasticity(activity_source, calcium_target, connectivity):

    growth_matrix = np.tile(activity_source,
                            (len(calcium_target), 1)).T  # matrix indicating how much each parallel fiber should grow
    growth_matrix += calcium_target + 0.01  # use broadcasting to add the calcium concentration and spontaneous growth
    # induced by climbing fibers to all granule cells activity vector (each duplicated row)
    growth_matrix = np.where(connectivity, growth_matrix, 0)
    growth_matrix = np.where(growth_matrix > threshold_plasticty, -growth_matrix, growth_matrix)
    growth_matrix = growth_matrix * delta_plasticity
    to_apply = lambda x: np.clip(x + growth_matrix, a_min=0,
                                 a_max=None)  # WARNING: could be problemetic if many events modify the synaptic weight.
    # If it is the case the clip should be outside of the event function. But here it is ok as this function is the only one modifying the
    return to_apply  # to add to parrallel fibers weights


def parallel_fiber_plasticity_2(activity_source, calcium_target, connectivity):
    growth_matrix = np.tile(activity_source,
                            (len(calcium_target), 1))  # matrix indicating how much each parallel fiber should grow$
    growth_matrix += calcium_target  # use broadcasting to add the calcium concentration
    # induced by climbing fibers to all granule cells activity vector (each duplicated row)
    growth_matrix = np.where(connectivity, growth_matrix, 0)
    growth_matrix = np.where(growth_matrix > threshold_plasticty, threshold_plasticty - growth_matrix, growth_matrix)
    growth_matrix = growth_matrix * delta_plasticity
    to_apply = lambda x: np.clip(x + growth_matrix, a_min=0,
                                 a_max=None)  # WARNING: could be problemetic if many events modify the synaptic weight.
    # If it is the case the clip should be outside of the event function. But here it is ok as this function is the only one modifying the
    return to_apply  # to add to parrallel fibers weights


def update_copy_state(cere_state_copy, cere_state):
    for neuron_pop in cere_state["neuron_pop"]:
        for local_variables in cere_state["neuron_pop"][neuron_pop]["local_variables"]:
            np.copyto(cere_state_copy["neuron_pop"][neuron_pop]["local_variables"][local_variables],
                      cere_state["neuron_pop"][neuron_pop]["local_variables"][local_variables])

        for synapse_pop in cere_state["synapse_pop"]:
            for local_variables in cere_state["synapse_pop"][synapse_pop]["local_variables"]:
                np.copyto(cere_state_copy["synapse_pop"][synapse_pop]["local_variables"][local_variables],
                          cere_state["synapse_pop"][synapse_pop]["local_variables"][local_variables])
