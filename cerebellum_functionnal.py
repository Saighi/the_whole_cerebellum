from neuron_pop import NeuronPop
from utils import *
import numpy as np
from copy import deepcopy

# Bunch of metavariables
divergence=3
input_size=100
DCN_size=100
IO_size=100
GC_size=100
PC_size=100
min_activity=0
max_activity=1

cere_info = dict()

cere_info["neuron_pop"]=dict()
cere_info["neuron_pop"]["DCN"]=dict()
cere_info["neuron_pop"]["DCN"]["size"]=DCN_size

cere_info["neuron_pop"]["IO"]=dict()
cere_info["neuron_pop"]["IO"]["size"]=IO_size

cere_info["neuron_pop"]["GC"]=dict()
cere_info["neuron_pop"]["GC"]["size"]=GC_size

cere_info["neuron_pop"]["PC"]=dict()
cere_info["neuron_pop"]["PC"]["size"]=PC_size

cere_info["neuron_pop"]["PT"]=dict()
cere_info["neuron_pop"]["PT"]["size"]=input_size

# create an activity vector variable for each region of the cerebellum (any local variable is considered a vector
# and a tuple is used to store the shape)
for key in cere_info["neuron_pop"]:
    cere_info["neuron_pop"][key]["local_variables"]=dict()
    cere_info["neuron_pop"][key]["local_variables"]["activity"] = (cere_info["neuron_pop"][key]["size"],1)
    pass

# Purkinje cells are a little bit special as we add a local variable called calcium to deal with synaptic plasticity

cere_info["neuron_pop"]["PC"]["local_variables"]["calcium"] = (cere_info["neuron_pop"]["PC"]["size"],1)

""" probability of connection based on the populations sizes 
and the divergence factor """

p_GC_PC = ((PC_size/GC_size)/GC_size)*divergence
p_PC_DCN = ((DCN_size/PC_size)/PC_size)*divergence
p_DCN_IO = ((IO_size/DCN_size)/DCN_size)*divergence
p_IO_PC = ((PC_size/IO_size)/DCN_size)*divergence
p_PT_GC = ((GC_size/input_size)/input_size)*divergence
p_PT_DCN = ((DCN_size/input_size)/input_size)*divergence
p_PT_IO = ((IO_size/input_size)/input_size)*divergence

cere_info["synapse_pop"]=dict()
cere_info["synapse_pop"]["GC_PC_syn"]=dict()
cere_info["synapse_pop"]["GC_PC_syn"]["i_o"]=("GC","PC")
cere_info["synapse_pop"]["GC_PC_syn"]["type"]="excitatory"
cere_info["synapse_pop"]["GC_PC_syn"]["p"]=p_GC_PC
cere_info["synapse_pop"]["PC_DCN_syn"]=dict()
cere_info["synapse_pop"]["PC_DCN_syn"]["i_o"]=("PC","DCN")
cere_info["synapse_pop"]["PC_DCN_syn"]["type"]="inhibitory"
cere_info["synapse_pop"]["PC_DCN_syn"]["p"]=p_PC_DCN
cere_info["synapse_pop"]["DCN_IO_syn"]=dict()
cere_info["synapse_pop"]["DCN_IO_syn"]["i_o"]=("DCN","IO")
cere_info["synapse_pop"]["DCN_IO_syn"]["type"]="inhibitory"
cere_info["synapse_pop"]["DCN_IO_syn"]["p"]=p_DCN_IO
cere_info["synapse_pop"]["IO_PC_syn"]=dict()
cere_info["synapse_pop"]["IO_PC_syn"]["i_o"]=("IO","PC")
cere_info["synapse_pop"]["IO_PC_syn"]["type"]="Undefined"
cere_info["synapse_pop"]["IO_PC_syn"]["p"]=p_IO_PC
cere_info["synapse_pop"]["PT_GC_syn"]=dict()
cere_info["synapse_pop"]["PT_GC_syn"]["i_o"]=("PT","GC")
cere_info["synapse_pop"]["PT_GC_syn"]["type"]="excitatory"
cere_info["synapse_pop"]["PT_GC_syn"]["p"]=p_PT_GC
cere_info["synapse_pop"]["PT_DCN_syn"]=dict()
cere_info["synapse_pop"]["PT_DCN_syn"]["i_o"]=("PT","DCN")
cere_info["synapse_pop"]["PT_DCN_syn"]["type"]="excitatory"
cere_info["synapse_pop"]["PT_DCN_syn"]["p"]=p_PT_DCN
cere_info["synapse_pop"]["PT_IO_syn"]=dict()
cere_info["synapse_pop"]["PT_IO_syn"]["i_o"]=("PT","IO")
cere_info["synapse_pop"]["PT_IO_syn"]["type"]="excitatory"
cere_info["synapse_pop"]["PT_IO_syn"]["p"]=p_PT_IO

for key in cere_info["synapse_pop"]:
    cere_info["synapse_pop"][key]["mean_weight"]=1/divergence
    cere_info["synapse_pop"][key]["local_variables"]=dict()
    input_pop = cere_info["synapse_pop"][key]["i_o"][0]
    input_pop_size = cere_info["neuron_pop"][input_pop]["size"]
    target_pop = cere_info["synapse_pop"][key]["i_o"][1]
    target_pop_size = cere_info["neuron_pop"][target_pop]["size"]
    cere_info["synapse_pop"][key]["local_variables"]["weight"] = (input_pop_size,target_pop_size)


cere_state = dict() # Will store all the vectors describing the state of the cerebellum at one given moment

cere_state["neuron_pop"]=dict()
cere_states_neurons_pointers = list() # a list of all the state variables to make it easier (and faster) to set all neuron activity state to zero
for key in cere_info["neuron_pop"]:
    cere_state["neuron_pop"][key]=dict()
    cere_state["neuron_pop"][key]["local_variables"]=dict()
    for key2 in cere_info["neuron_pop"][key]["local_variables"]:
        shape = cere_info["neuron_pop"][key]["size"]
        cere_state["neuron_pop"][key]["local_variables"][key2]=np.zeros(shape)
        cere_states_neurons_pointers.append(cere_state["neuron_pop"][key]["local_variables"][key2])

cere_state["synapse_pop"]=dict()
for key in cere_info["synapse_pop"]:
    cere_state["synapse_pop"][key]=dict()
    cere_state["synapse_pop"][key]["local_variables"]=dict()
    p = cere_info["synapse_pop"][key]["p"]
    input_pop = cere_info["synapse_pop"][key]["i_o"][0]
    input_pop_size = cere_info["neuron_pop"][input_pop]["size"]
    target_pop = cere_info["synapse_pop"][key]["i_o"][1]
    target_pop_size = cere_info["neuron_pop"][target_pop]["size"]
    mean_weight = cere_info["synapse_pop"][key]["mean_weight"]
    cere_state["synapse_pop"][key]["local_variables"]["weight"]= gene_w_mat(p,input_pop_size,target_pop_size,mean_weight)

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
new_event["args_paths"]=(("neuron_pop","IO","local_variables","activity"),
            ("synapse_pop","IO_PC_syn","local_variables","weight")) 
new_event["output"] = ("neuron_pop","PC","local_variables","calcium")
new_event["function"]=default_excitatory_behavior 
cere_events.append(new_event)
new_event=dict()
# correspond to the actual plasticity mecanism depending on granule cells 
# activity and calcium transients induced by IO activity in Purkinje cells
new_event["args_paths"]=(("neuron_pop","GC","local_variables","activity"),
            ("neuron_pop","PC","local_variables","calcium"),
            ("synapse_pop","GC_PC_syn","local_variables","weight"))
new_event["output"] = ("synapse_pop","GC_PC_syn","local_variables","weight")
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
every variables is directly accessible via event pointers and dictionaries "leafs" are directly modifiable.
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

"""Here is the main loop where the algorithm is going on"""
"""
cere_state["neuron_pop"]["PT"]["local_variables"]["activity"][:]=np.ones(100)#Try to put some value for the network inputs
cere_state_pre["neuron_pop"]["PT"]["local_variables"]["activity"][:]=np.ones(100)
#np.copyto(event["output"],event["function"](*event["args_tuple"]))
print(cere_events[3])
print(event_pointers[3])
test_event= event_pointers[3]
output_event = test_event["function"](*test_event["args_tuple"])
np.copyto(test_event["output"],output_event(test_event["output"]))
print("GC_activity")
print(cere_state["neuron_pop"]["GC"]["local_variables"]["activity"])
update_copy_state(cere_state_pre,cere_state)
print(cere_state_pre["neuron_pop"]["GC"]["local_variables"]["activity"])
for neuron_state in cere_states_neurons_pointers:
        neuron_state.fill(0)
print("to zero GC actual state?")
print(cere_state["neuron_pop"]["GC"]["local_variables"]["activity"])
print("to one PT actual state?")
print(cere_state["neuron_pop"]["PT"]["local_variables"]["activity"])
output_event = test_event["function"](*test_event["args_tuple"])
#print("what is in our arguments ??? NO refresh !")
#print(event_pointers[3])

print("output of the event :")
print(np.mean(output_event(test_event["output"])))
"""

nb_time_steps = 10
for t in range(nb_time_steps):
    update_copy_state(cere_state_pre,cere_state)
    for neuron_state in cere_states_neurons_pointers:
        neuron_state.fill(0)
    cere_state["neuron_pop"]["PT"]["local_variables"]["activity"][:]=np.ones(100)#Try to put some value for the network inputs
    for event in event_pointers:
        np.copyto(event["output"],event["function"](*event["args_tuple"])(event["output"]))

print(cere_state["neuron_pop"])
#print(event_pointers[3])