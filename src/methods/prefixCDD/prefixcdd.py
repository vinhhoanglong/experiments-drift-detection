import uuid
import pm4py
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from tqdm import tqdm
from river import drift
from pm4py.objects.log.obj import EventLog




def extract_caseid_act(log: EventLog) -> List[Tuple[str, str]]:
    """
    """
    caseid_act_pair = []
    for trace in log:
        case_id = trace.attributes.get('concept:name', str(id(trace)))

        for event in trace:
            act = event.get('concept:name', 'Unknown Activity')
            caseid_act_pair.append((case_id, act))

    return caseid_act_pair


def _group_event_by_key(events: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    """
    grouped = defaultdict(list)
    for case_id, act in events:
        grouped[case_id].append(act)
    return grouped






class PrefixNode:
    """
    A node in prefix tree store activity label, freq, children
    """
    def __init__(self, activity: str, parent=None):
        self.id = uuid.uuid4()
        self.activity = activity
        self.freq = 0
        self.children:Dict[str, 'PrefixNode'] = {}
        self.parent = parent

    def add_child(self, activity: str):
        """
        Add or return an existing child node for  the given activity
        """
        if activity not in self.children:
            self.children[activity] = PrefixNode(activity, self)
        return self.children[activity]

class PrefixTree:

    """
    representing the collection of node ~ as the trace execution
    """
    def __init__(self):
        self.root = PrefixNode("root")

    def add_trace(self, trace:List[str]):
        node = self.root
        for activity in trace:
            node = node.add_child(activity)
        node.freq += 1

    def get_node_fred(self) -> Dict[str, int]:
        results = defaultdict(int)

        def traverse(node):
            for child in node.children.values():
                results[child.activity] += child.freq
                traverse(child)
        traverse(self.root)
        return dict(results)



def compute_tree_distance(t1: PrefixTree, t2: PrefixTree) -> float:
    f1 = t1.get_node_fred()
    f2 = t2.get_node_fred()
    all_acts = set(f1.keys()).union(f2.keys())
    diff = 0
    for act in all_acts:
        diff += (f1.get(act, 0) - f2.get(act, 0)) ** 2
    return np.sqrt(diff)
def build_tree(events: List[Tuple[str, str]]) -> PrefixTree:
    """
    """
    tree = PrefixTree()
    cases = _group_event_by_key(events)
    for trace in cases.values():
        tree.add_trace(trace)
    return tree




class PrefixCDD:
    """
    """
    def __init__(self, nums_tree: int = 12, events_per_tree: int = 500, show_progress_bar: bool = True):
        """
        """
        self.nums_tree = nums_tree
        self.events_per_tree = events_per_tree
        self.show_progress_bar = show_progress_bar
        self.trees:List[PrefixTree] = []
        self.current_events:List[Tuple[str, str]] = []
        self.adwin = drift.ADWIN(delta = 5)
        self.detected_drifts:List[int] = []
        self.total_events = 0
        self.drift_info = []

    def add_event(self, case_id:str, act:str):
        """
        """
        self.current_events.append((case_id, act))
        self.total_events += 1

        if len(self.current_events) >= self.events_per_tree:
            new_tree = build_tree(self.current_events)
            self.trees.append(new_tree)
            self.current_events = []

            if len(self.trees) > self.nums_tree:
                self.trees.pop(0)

            if len(self.trees) >= 2:
                distance = compute_tree_distance(self.trees[-1], self.trees[-2])
                self._update_drift_detector(distance, case_id, act)
                
    def _update_drift_detector(self, distance:float, case_id:str, act:str):
        """
        """
        self.adwin.update(distance)
        if self.adwin.drift_detected:
            drift_index = case_id
            self.detected_drifts.append(drift_index)
            drift_event_index = self.total_events
            
            self.drift_info.append({
                                   "tree": drift_index,
                                   "event_index": drift_event_index,
                                   "case_id": case_id,
                                   "activity": act,
                                   "distance": distance
                                   })

            print(f"[Drift] Tree {drift_index}, Event {drift_event_index}, "
                  f"Case {case_id}, Activity '{act}', δ={distance:.4f}")


    def process_stream(self, event_stream: List[Tuple[str, str]]):
        """
        """
        if self.show_progress_bar:
            iterator = tqdm(event_stream, desc= "process stream")
        else:
            iterator = event_stream

        for case_id, act in iterator:
            self.add_event(case_id, act)

        return self.detected_drifts


def __main__():
    data_path = 'data/bose_log.xes'
    log = pm4py.read_xes(data_path)
    cid_act_pair = extract_caseid_act(log)
    evnts = _group_event_by_key(cid_act_pair)
    event_stream = [(case_id, act) for case_id, acts in evnts.items() for act in acts]
    cdd = PrefixCDD(nums_tree = 12, events_per_tree=300)
    cdd.process_stream(event_stream)

if __name__=="__main__":
    __main__()




