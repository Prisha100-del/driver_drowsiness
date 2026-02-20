import numpy as np
 
def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)
 
def compute_ear(eye):
    # eye: array of 6 (x, y) points
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear