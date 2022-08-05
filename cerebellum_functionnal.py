from utils import *
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt 

def whole_model_sim(cere_info,input_dict):
    """Simulation of the microcircuitry of the Cerebellum

    Keyword arguments:
    params -- a dictionary with all the parameters of the model
    """

    cere_state = dict() # Will store all the vectors describing the state of the cerebellum at one given moment
    cere_state["neuron_pop"]=dict()
    for key in cere_info["neuron_pop"]:
        cere_state["neuron_pop"][key]=dict()
        cere_state["neuron_pop"][key]["local_variables"]=dict()
        for key2 in cere_info["neuron_pop"][key]["local_variables"]:
            shape = cere_info["neuron_pop"][key]["size"]
            cere_state["neuron_pop"][key]["local_variables"][key2]=np.zeros(shape)

    cere_state["synapse_pop"]=dict() 
    for key in cere_info["synapse_pop"]:
        cere_state["synapse_pop"][key]=dict()
        cere_state["synapse_pop"][key]["local_variables"]=dict()
        nb_connect = cere_info["synapse_pop"][key]["nb_connect"]
        input_pop = cere_info["synapse_pop"][key]["i_o"][0]
        input_pop_size = cere_info["neuron_pop"][input_pop]["size"]
        target_pop = cere_info["synapse_pop"][key]["i_o"][1]
        target_pop_size = cere_info["neuron_pop"][target_pop]["size"]
        mean_weight = cere_info["synapse_pop"][key]["mean_weight"]
        minimum_connect= cere_info["synapse_pop"][key]["minimum_connect"]
        if key == "IO_PC":#Dealing with IO connectivity and weights
            weight_matrice,connectivity = gene_w_exclusive(int(target_pop_size/input_pop_size),input_pop_size,target_pop_size,mean_weight)
            cere_state["synapse_pop"][key]["local_variables"]["weight"]= weight_matrice
            cere_state["synapse_pop"][key]["local_variables"]["connectivity"]= connectivity
        else: 
            weight_matrice,connectivity = gene_w_inclusive(nb_connect,input_pop_size,target_pop_size,mean_weight,minimum_connect)
            cere_state["synapse_pop"][key]["local_variables"]["weight"]= weight_matrice
            cere_state["synapse_pop"][key]["local_variables"]["connectivity"] = connectivity

    cere_events = list() # Will store all actions needed to account for the algorithm, mainly synaptic transmission 

    """Synapses are the main source of events in the model, their input/output and their type excitatory/inhibitory 
    are used to derive the kind of function and which state variable is affected"""
    for key in cere_info["synapse_pop"]:
        new_event=dict()
        source = cere_info["synapse_pop"][key]["i_o"][0]
        target = cere_info["synapse_pop"][key]["i_o"][1]
        type = cere_info["synapse_pop"][key]["type"]
        if type=="excitatory":
            new_event["args_paths"]=(("neuron_pop",source,"local_variables","activity"),
                ("synapse_pop",key,"local_variables","weight"))
            new_event["output"] = ("neuron_pop",target,"local_variables","activity")
            new_event["function"]=default_excitatory_behavior
            cere_events.append(new_event)
        elif type=="inhibitory":
            new_event["args_paths"]=(("neuron_pop",source,"local_variables","activity"),
                ("synapse_pop",key,"local_variables","weight"))
            new_event["output"] = ("neuron_pop",target,"local_variables","activity")
            new_event["function"]=default_inhibitory_behavior
            cere_events.append(new_event)

    """Some events are custom and correspond to the specific behavior of the Cerebellum"""

    # correspond to the presynaptic increase in calcium leading to plasticity
    new_event=dict()
    new_event["args_paths"]=(("neuron_pop","IO","local_variables","activity"),
                ("synapse_pop","IO_PC","local_variables","weight")) 
    new_event["output"] = ("neuron_pop","PC","local_variables","calcium")
    new_event["function"]=default_excitatory_behavior 
    cere_events.append(new_event)
    # correspond to the actual plasticity mecanism depending on granule cells 
    # activity and calcium transients induced by IO activity in Purkinje cells
    new_event=dict()
    new_event["args_paths"]=(("neuron_pop","GC","local_variables","activity"),
                ("neuron_pop","PC","local_variables","calcium"),
                ("synapse_pop","GC_PC","local_variables","connectivity"))
    new_event["output"] = ("synapse_pop","GC_PC","local_variables","weight")
    new_event["function"]=parallel_fiber_plasticity 
    cere_events.append(new_event)

    """It's time to simulate the cerebellum by making each event append.
    Input data for each events will be from the t-1 cerebellum and output 
    data will give the states of t cerebellum.
    We then need to make a copy of the previous state of the cerebellum in
    order to not overwrite the t-1 states as we are computing the t states.
    This way each events can be made in an arbitrary order"""

    cere_state_pre = deepcopy(cere_state)

    """This whole section is about getting pointers of input, output and functions corresponding to state vectors and
    transformations for each events, then we don't need to loop through the dictionaries during the simulation 
    every variables is directly accessible via event pointers and dictionaries of "leafs" are directly modifiable.
    IMPORTANT: input are from the previous cere_state (cere_state_pre) and output from the actual cere_state,
    this way the event order is not important as we are modifying a copy of the cerebellum state. """
    event_pointers = list() # will contain dictionaries for each events containing pointers toward inputs outputs
    # and functions used in the context of that event

    for event in cere_events:
        event_pointers.append(dict())
        args_list_pointers = list()

        for arg_path in event["args_paths"]:
            branch = cere_state_pre # will branche in cere_state_pre dictionary, input are from the previous state !
            for step in arg_path: # at the end of this loop branch is a leaf corresponding to a state vector (one input)
                branch = branch[step]
            args_list_pointers.append(branch)

        event_pointers[-1]["args_tuple"]=args_list_pointers

        output_path = event["output"] # as there is only one output for each function we don't have to loop over output.
        branch = cere_state # will branche in cere_state dictionary, output modify the actual state !
        for step in output_path: # at the end of this loop branch is a leaf corresponding to a state vector (the output)
            branch = branch[step]

        event_pointers[-1]["output"]=branch

        event_pointers[-1]["function"]=event["function"]

    """Main Loop"""
    t=0
    while True:
        update_copy_state(cere_state_pre,cere_state)

        for neuron_pop in cere_state["neuron_pop"]:
            for local_variable in cere_state["neuron_pop"][neuron_pop]["local_variables"]:
                cere_state["neuron_pop"][neuron_pop]["local_variables"][local_variable][:]=input_dict[neuron_pop][local_variable][t]

        for event in event_pointers:
            np.copyto(event["output"],event["function"](*event["args_tuple"])(event["output"]))
    
        for neuron_pop in cere_state["neuron_pop"]:
            cere_state["neuron_pop"][neuron_pop]["local_variables"]["activity"][:]=np.clip(cere_state["neuron_pop"][neuron_pop]["local_variables"]["activity"],a_min = 0,a_max=None)

            # if neuron_pop!="PT":
            #     cere_state["neuron_pop"][neuron_pop]["local_variables"]["activity"][:]=(cere_state["neuron_pop"][neuron_pop]["local_variables"]["activity"]-np.mean(cere_state["neuron_pop"][neuron_pop]["local_variables"]["activity"]))+1
            #     cere_state["neuron_pop"][neuron_pop]["local_variables"]["activity"][:]=np.clip(cere_state["neuron_pop"][neuron_pop]["local_variables"]["activity"],a_min = 0,a_max=None)
          
        yield cere_state
        t+=1
      

    # print([(name,np.mean(activity["local_variables"]["activity"])) for name,activity in cere_state["neuron_pop"].items()])
    # print(('calcium_pc',np.mean(cere_state["neuron_pop"]["PC"]["local_variables"]["calcium"])))
    # print(cere_state["neuron_pop"]["PC"]["local_variables"]["calcium"])
    # print([(name,np.min(activity["local_variables"]["activity"]),np.max(activity["local_variables"]["activity"])) for name,activity in cere_state["neuron_pop"].items()])
    #print(event_pointers[3]) """
    #print(np.min(cere_state["synapse_pop"]["GC_PC"]["local_variables"]["weight"]))
   
