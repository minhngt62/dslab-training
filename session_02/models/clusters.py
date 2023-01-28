from typing import List

class Member:
    def __init__(self, r_d, label = None, doc_id = None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id

class Cluster:
    def __init__(self):
        self._centroid = None 
        self._members: List[Member] = []
    
    def reset_members(self):
        '''
        Empty out cluster
        '''
        self._members = []
    
    def set_centroid(self, new_centroid: Member):
        '''
        Set new centroids for cluster
        '''
        self._centroid = new_centroid
    
    def add_members(self, new_member: Member):
        '''
        Add a new data point to cluster
        '''
        self._members.append(new_member)