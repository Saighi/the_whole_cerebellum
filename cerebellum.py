import numpy as np
from synapse_pop import SynapsePop
from neuron_pop import NeuronPop
from utils import *


class Cerebellum:
    """ The whole cerebellum

    Parameters
    ----------
    input_size: int
        Size of the input vector corresponding to the number of neurons in pontine population
    divergence: float, optional (Default: 10)
        the divergence factor directly influence the connectivity probability between all areas.
        A divergence factor of X mean that each neuron of the target population receives 
        in average X synapses of the source population.
        A too high divergence factor will induce an uniformisation of the information carried by
        each vectors. 
    DCN_size: int
    IO_size: int
    GC_size: int
    PC_size: int
    """

    def __init__(self,
                 divergence=3,
                 input_size=100,
                 DCN_size=100,
                 IO_size=100,
                 GC_size=100,
                 PC_size=100):

        self.index=dict() 
        """The INDEX is the central data structure of this simulation, will gather name/object correspondance
        for each neuron population and synapses. We use an indexe instead of obect references 
        inside other objects (such as a synapse keeping a reference of a neuron population) 
        to deal better with copies of the cerebellar state."""
  

        self.scale = 1
        self.DCN = NeuronPop("DCN",DCN_size)
        self.index["DCN"]=self.DCN
        self.IO = NeuronPop("IO",IO_size)
        self.index["IO"]=self.DCN
        self.GC = NeuronPop("GC",GC_size)
        self.index["GC"]=self.DCN
        self.PC = NeuronPop("PC",PC_size)
        self.index["PC"]=self.DCN

        self.PC.add_local_variable("calcium", np.zeros(PC_size))
        #TODO input size should be artificially increased by copying part of the input vector
        self.PT = NeuronPop("PT",input_size)

        """ probability of connection based on the populations sizes 
        and the divergence factor """

        p_GC_PC = ((PC_size/GC_size)/GC_size)*divergence
        p_PC_DCN = ((DCN_size/PC_size)/PC_size)*divergence
        p_DCN_IO = ((IO_size/DCN_size)/DCN_size)*divergence
        p_IO_PC = ((PC_size/IO_size)/DCN_size)*divergence
        p_PT_GC = ((GC_size/input_size)/input_size)*divergence
        p_PT_DCN = ((DCN_size/input_size)/input_size)*divergence
        p_PT_IO = ((IO_size/input_size)/input_size)*divergence


        self.GC_PC_syn = SynapsePop(
            "GC_PC_syn",self.GC, self.PC, p_GC_PC,1/divergence,SynapseType.CUSTOM)
        self.index["GC_PC_syn"]=self.GC_PC_syn
        self.GC_PC_syn.define_behavior(parallel_fiber_behavior)

        self.PC_DCN_syn = SynapsePop(
            "PC_DCN_syn",self.PC, self.DCN, p_PC_DCN,1/divergence,SynapseType.INHIBITORY)
        self.index["PC_DCN_syn"]=self.PC_DCN_syn

        self.DCN_IO_syn = SynapsePop(
            "DCN_IO_syn",self.DCN, self.IO, p_DCN_IO,1/divergence,SynapseType.INHIBITORY)
        self.index["DCN_IO_syn"]=self.DCN_IO_syn

        self.IO_PC_syn = SynapsePop(
            "IO_PC_syn",self.IO, self.PC, p_IO_PC, 1/divergence, SynapseType.CUSTOM)
        self.index["IO_PC_syn"]=self.IO_PC_syn
        self.IO_PC_syn.define_behavior(climbing_fiber_behavior)

        self.PT_GC_syn = SynapsePop(
            "PT_GC_syn",self.PT, self.GC, p_PT_GC,1/divergence,SynapseType.EXCITATORY)
        self.index["PT_GC_syn"]=self.PT_GC_syn
        
        self.PT_DCN_syn = SynapsePop(
            "PT_DCN_syn",self.PT, self.DCN, p_PT_DCN,1/divergence,SynapseType.EXCITATORY)
        self.index["PT_DCN_syn"]=self.PT_DCN_syn

        self.PT_IO_syn = SynapsePop(
            "PT_IO_syn",self.PT, self.IO, p_PT_IO,1/divergence,SynapseType.EXCITATORY)
        self.index["PT_IO_syn"]=self.PT_IO_syn
    
    

        
