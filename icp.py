import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


def gen_lidar():
    pass

class ICP:
    def __init__(self):
        pass

    def match(self, P_s, P_sp, C_guess=None, r_guess=None):
        """ Find optimal rotation and translation between two point-clouds.

        Parameters
        ----------
            P_s: 3xN Numpy array.
            P_sp: 3xM Numpy array.
        """

        # 1. Find an initial guess for rotation and translation [C_sps, r_s].

        # 2. Associate each point in P_sp with nearest point in P_s.

        # 3. Solve for optimal transformation...

            # 3.1 Compute centroids of each point-cloud.

            # 3.2 Compute `W` matrix, capturing "spread" of point-clouds.

            # 3.3 Use SVD of `W` to compute optimal rotation.

            # 3.4 Using optimal rotation, solve for optimal translation.
