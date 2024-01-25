import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import distance_matrix
import utils

class model():

    def __init__(self, lengths, ends):
        self.lengths = lengths # N x N array of natural lengths. Infinite if no edge exists between two nodes. Presumably, this is symmetric.
        self.ends = ends
        self.status = None
        self.optimal_path = None
        self.optimal_distance = None

    def solve(self, iterations, separation_distance, stiffness, stepsize, gravity = -9.81, mass = 1, method = 'Euler') -> None:

        N = self.lengths.shape[0]

        # Set Initial conditions
        current_locations = np.zeros(N,2)
        
        for iteration in range(iterations):
            
            # Move the second endpoint separation_distance along the x axis
            current_locations[self.ends[1], 0] += separation_distance

            # Reset force matrix to only include gravitational force except for ends (because we're artificially holding them up)
            forces = np.tile(np.array([0, mass * gravity]), N).reshape([N, 2])
            
            # Potential nodes along potentially optimal path (ie. where edges are stretched beyond their natural lengths)
            edge_candidates = np.argwhere(distance_matrix(current_locations, current_locations) > self.lengths)
        
            if edge_candidates.size == 0:
                current_locations += stepsize / mass * forces
                # In this specific order to multiply numpy array by scalar elementwise once
                continue
        
            node_candidates = np.unique(edge_candidates)
            # First check which strings have exceeded their natural lengths and check if stretched strings form path from end to end
            if (self.ends[0] in node_candidates) & (self.ends[1] in node_candidates):

                mod_ends = np.argwhere((node_candidates == self.ends[0]) | (node_candidates == self.ends[1])).tolist()
                dist_matrix, predecessors = shortest_path(csr_matrix(self.lengths[np.ix_(node_candidates, node_candidates)]), return_predecessors = True)

                if predecessors[mod_ends[0], mod_ends[1]] != -9999: # i.e. a path has been found

                    # Reconstruct path from predecessors
                    optimal_path = [mod_ends[0]]
                    k = mod_ends[0]
                    while predecessors[mod_ends[1], k] != -9999:
                        optimal_path.append(predecessors[mod_ends[1], k])
                        k = predecessors[mod_ends[1], k]

                    self.optimal_path = np.vectorize(dict(enumerate(node_candidates)).get)(optimal_path)
                    self.optimal_distance = dist_matrix[mod_ends[0], mod_ends[1]]
                    self.status = 0
                    return

            # If no path found, then calculate the forces of those which have exceeded their natural lengths (edge_candidates) if no valid path exists yet
            edge_candidates = list({set(pair) for pair in np.unique(edge_candidates, 0)}) # Should result in a list of sets, the sets being the combinations of candidate nodes (which should not be duplicated as to prevent unecessary computation)
            edge_candidates = [list(pair) for pair in edge_candidates] # Gives us list of candidate edges without duplicates, because symmetry
            for combo in edge_candidates:
                
                forces[combo[0], 0], forces[combo[0], 1], forces[combo[1], 0], forces[combo[1], 1] = utils.hooke(current_locations[combo[0], 0], current_locations[combo[0], 1], current_locations[combo[1], 0], current_locations[combo[1], 1], self.lengths[combo[0], combo[1]], stiffness)


            forces[self.ends[0], 0] = 0
            forces[self.ends[0], 1] = 0
            forces[self.ends[1], 0] = 0
            forces[self.ends[1], 1] = 0

            current_locations += stepsize / mass * forces # Divide by `mass` first to modify `forces` by only one scalar operation

        self.status = 0