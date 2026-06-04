"""
Main FSDP Recognition Logic 

=> Pre Forward (layer X) in der CPU finden => Alle GPU Kernel in FSDPUnit.all_gather einfügen 


"""

from typing import Dict, List, Optional, Tuple, Set, Iterator, Any
from enum import Enum

from trace_parser import LogicalOperation

"""
Unterscheide nach Steps 
"""


class FSDP: 
    def __init__(self):
        units: list[FSDPUnit] = []
        

class FSDPUnit: 
    '''Unterscheide GPU User Annotation von Kernels'''
    '''Potentiell mehr als nur diese Phasen (Optimizer, Unit -1, NCCL vs Copy Out)'''
    def __init__(self):
        all_gather_fwd: list = []
        all_gather_bwd: list = []
        reduce_scatter: list = []
        fwd_compute: list = []
        bwd_compute: list = []

class StandardFSDPDetector: 
    def __init__(self):
        pass

    def _detect_all_gather_fwd(self, roots: List[LogicalOperation], fsdp_units: FSDP):
        '''
        Versuch 1: Finde fsdp::pre_forward und nimm alle Kernel Children 
        Versuch 2: Finde GPU User Annotation FSDP::all_gather(layer X)
                    => Nimm alle Kinder 
                   Finde fsdp::copy_out(Layer X)
                   => NCCL Event zwischen diesen beiden 
                   => Vergleiche welches NCCL Event in der Unit davor schon war 

        + Return Copy Out Indeces, falls es sich anbietet 
        
        '''
        pass

    def _detect_all_gather_bwd(self, roots: List[LogicalOperation], fsdp_units: FSDP):
        '''
        Analog 
        '''
        pass

    def _detect_fwd_cmp(self, roots: List[LogicalOperation], fsdp_units: FSDP):
        '''
        1 - Finde den Compute Stream 
        2 - Einmal iterieren 
        Alle Kernel zwischen CopyOut(layer X) und X+1 sind FWD_CMP X 
        Spezialfall: Letzer Forward Comp vor Loss Berechnung 
        '''
        pass

    def _detect_bwd_cmp(self, roots: List[LogicalOperation], fsdp_units: FSDP): 
        '''
        1 - Finde den Compute Stream
        2 - Regel 
            CopyOut(layer X) inklusive post_backward 
            => backward_reduce zu Reduce Scatter 
        '''
        pass

    def _detect_reduce_scatter(self, roots: List[LogicalOperation], fsdp_units: FSDP): 
        '''
        Versuch 1: Analog 
        Versuch 2: Post Backward Reduce 

        + Return Copy Out Indeces, falls es sich anbietet 
        '''
        pass


    def extract_fsdp_phases(self, roots: List[LogicalOperation]) -> FSDP:
        fsdp_units = FSDP()
        all_events = None 

        self._detect_all_gather_fwd(all_events, fsdp_units)
        self._detect_fwd_cmp(all_events, fsdp_units)
        self._detect_all_gather_bwd(all_events, fsdp_units)
        self._detect_reduce_scatter(all_events, fsdp_units)
        self._detect_bwd_cmp(all_events, fsdp_units)

        return fsdp_units