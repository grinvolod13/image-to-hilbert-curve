import numpy as np
import hilbert

def image_to_hilbert_curve(img: np.ndarray) -> np.ndarray:
    hilbert_order = int(np.ceil(np.log2(max(img.shape[:2]))))

    # Upsize image to nearest biggest curve order (h = w = 2^hilbert_order)
    bigger_image = np.zeros([2**hilbert_order, 2**hilbert_order, *(img.shape[2:])])
    bigger_image[:img.shape[0], :img.shape[1]] = img 
    
    return np.array([bigger_image[i, j] for (i, j) in hilbert.decode(range(2**(2*hilbert_order)), 2,  hilbert_order)])