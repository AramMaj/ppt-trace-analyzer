"""
Bottleneck und Metriken 

Rückgabe 
=> Vorherige JSON + Erweiterungen über User Annotations und Bottlenecks 
=> Textueller Report
    < Bottleneck Freitext Report > 
    => Metriken für gesamte Unit    
    => Metriken pro Unit 
    => Unit Auflistung 
    => Chronlogical  
    
"""

from typing import Dict, List, Optional, Tuple, Set, Iterator, Any

from trace_parser import LogicalOperation

"""

"""
class Metrics: 
    """Altes Anpassen und neue Datenstruktur ausnutzen"""
    """Metriken aus dem Discord"""

class Bottlenecks: 
    """Schwellenwerte ausgehend von Metriken"""

class Report: 

    def generate_report(): 
        """Schreibt Report.txt und die JSON mit Makern"""
        pass

    def get_fsdp_timeline_aggregated_string(agg: Dict[str, Dict[str, float]]) -> str:
        """Kopieren"""
        pass

    def get_fsdp_chronological_timeline(roots: List[LogicalOperation]) -> str:
        """Kopieren"""
        pass