import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


class ICP:
    def __init__(self):
        self.neighbors = None
        self.C_best = None
        self.r_best = None
        self.last_loss = np.inf
        self.last_last_loss = np.inf

    def reset_state(self):
        """ Reset state variables."""
        self.neighbors = None
        self.C_best = None
        self.r_best = None
        self.last_loss = np.inf
        self.last_last_loss = np.inf

    def match(self, P_s, P_sp, max_iter=50, loss_threshold=0.25, distance_threshold=1.5):
        """ Find optimal rotation and translation between two point-clouds.

        Parameters
        ----------
            P_s: 3xN Numpy array.
            P_sp: 3xM Numpy array.
        """

        # self.reset_state()
        C_init = np.identity(3)
        r_init = np.zeros((3,1))

        no_improvement_counter = 0

        for i in range(max_iter):
            # 1. Find an initial guess for rotation and translation [C_sps, r_s].
            if self.C_best is not None:
                C_init = self.C_best
            if self.r_best is not None:
                r_init = self.r_best

            # 1.1 Apply initial transformation to P_s.
            P_s_for_matching = C_init @ (P_s - r_init)
            # print(1.1, P_s_for_matching.shape)

            # 2. Associate each point in P_sp with nearest point in P_s.
            # 2.1 Create NN object for finding neighbors.
            self.neighbors = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(P_s_for_matching.T)
            
            # 2.2 Find neighbors using NN object.
            point_pairs = self.nearest_point_pairs(P_s.T, P_sp.T, distance_threshold)
            # print(2.2, point_pairs.shape)
            try:
                P_s_paired = point_pairs[:,0,:].T
                P_sp_paired = point_pairs[:,1,:].T
                # print(2.2, P_s_paired.shape, P_sp_paired.shape)
            except IndexError:
                # print("Skipping due to IndexError")
                return False

            # 3. Solve for optimal transformation...
            self.update_best_guess(P_s_paired, P_sp_paired)
            # print(3, self.C_best, self.r_best) 

            # 4. Compute error, check for convergence.
            loss = self.compute_error(P_s_paired, P_sp_paired)
            # print(4, loss)
            if (loss >= self.last_loss or loss >= self.last_last_loss):
                no_improvement_counter += 1
                distance_threshold -= 0.1

            if no_improvement_counter > 25:
                # print(F"Did not converge, loss {loss}, Point Pairs: {point_pairs.shape}")
                return True

            if loss < loss_threshold:
                # print(F"Converged at {i}, with loss {loss}, Point Pairs: {point_pairs.shape}")
                return True

            self.last_last_loss = self.last_loss
            self.last_loss = loss

    def nearest_point_pairs(self, P_src, P_dst, distance_threshold=0.5, depth=0):
        """ Get neighbors of `self.neighbors` with `P_dst`.

        Parameters
        ----------
            P_src: Nx3 array of points.
            P_dst: Mx3 array of points.

        Returns
        -------
            closest_point_pairs: List of paired points.
            # distances: Euclidean distances between points P_dst and P_src.
            # indices: Indices of P_src corresponding with distance between P_dst.
        """

        closest_point_pairs = []  # list of point correspondences for closest point rule

        distances, indices = self.neighbors.kneighbors(P_dst)
        # print("Point Pair Shape", distances.shape, indices.shape)
        # print("NPP", P_src.shape, P_dst.shape)
        for nn_index in range(len(distances)):
            if distances[nn_index][0] < distance_threshold:
                # print(nn_index)
                pair = [P_src[indices[nn_index][0]], P_dst[nn_index]]
                # print("NPP", pair)
                closest_point_pairs.append(pair)
        # print("NPP", np.array(closest_point_pairs).shape)
        if len(closest_point_pairs) == 0 and depth < 10:
            return self.nearest_point_pairs(P_src, P_dst, distance_threshold=distance_threshold+0.5, depth=depth+1)

        return np.array(closest_point_pairs)

    def update_best_guess(self, P_s_paired, P_sp_paired):
        """ Step 3 in update loop."""

        # 3.1 Compute centroids of each point-cloud.
        # print(3.1, P_s.shape, P_sp.shape)
        mean_P_s = np.mean(P_s_paired, axis=1).reshape((3,1))
        mean_P_sp = np.mean(P_sp_paired, axis=1).reshape((3,1))
        # print(3.1, mean_P_s.shape, mean_P_sp.shape)

        # 3.2 Compute `W` matrix, capturing "spread" of point-clouds.
        # print(3.2, P_s.shape, P_sp.shape)
        # print(3.2, ((P_s - mean_P_s)).shape)
        W = ((P_s_paired - mean_P_s) @ (P_sp_paired - mean_P_sp).T) / P_s_paired.shape[1]
        # print(3.2, W.shape)

        # 3.3 Use SVD of `W` to compute optimal rotation.
        U, S, V_t = np.linalg.svd(W)
        # print(3.3, U.shape, S.shape, V_t.shape)
        C_sp_s = U @ np.diag([1, 1, np.linalg.det(U) * np.linalg.det(V_t.T)]) @ V_t

        # 3.4 Using optimal rotation, solve for optimal translation.
        r_s = mean_P_s - (C_sp_s @ mean_P_sp)
        self.C_best = C_sp_s
        self.r_best = r_s

    def compute_error(self, P_s_paired, P_sp_paired):
        loss_acc = 0
        E = self.C_best.T @ (P_s_paired - self.r_best) - P_sp_paired
        for i in range(E.shape[1]):
            loss_acc += abs(E[:,i].T @ E[:,i])

        return loss_acc


if __name__ == "__main__":
    from data_manager import read_into_documents
    import time

    icp = ICP()
    np.random.seed(12345)

    sine = np.sin(np.radians(8))
    cosine = np.cos(np.radians(8))
    rot = np.array([[cosine, -sine], [sine, cosine]])
    trans = np.array([[0.5], [0.25]])

    pts = np.arange(100).reshape(2,50) / 10
    pts_prime = rot @ (pts + trans)
    zeros = np.zeros((1,50))
    pts = np.vstack([pts, zeros])
    pts_prime = np.vstack([pts_prime, zeros])

    icp.match(pts, pts_prime)

    plt.plot(pts[0,:], pts[1,:], 'b+', label="Source")
    plt.plot(pts_prime[0,:], pts_prime[1,:], 'g+', label="Source Prime")

    pts_prime_to_plot = icp.C_best.T @ (pts - icp.r_best)
    plt.plot(pts_prime_to_plot[0,:], pts_prime_to_plot[1,:], 'r+', label="Destination")
    plt.legend()
    plt.show()
