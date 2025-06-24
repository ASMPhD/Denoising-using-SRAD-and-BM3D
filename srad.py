import numpy as np

def srad_denoising(image, num_iter=100, delta_t=0.125, q0_squared=1e-10):
    img = image.copy()
    for i in range(num_iter):
        # Estimate gradients
        Ix = np.gradient(img, axis=1)
        Iy = np.gradient(img, axis=0)
        grad_sq = Ix**2 + Iy**2

        # Laplacian
        lap = cv2.Laplacian(img, cv2.CV_64F)

        # Diffusion coefficient
        num = (0.5 * grad_sq - (1.0/16.0) * lap**2)
        den = (1.0 + (1.0/4.0) * grad_sq)
        q = num / (den + 1e-10)

        c = 1.0 / (1.0 + ((q - q0_squared) / (q0_squared + 1e-10))**2)

        # Apply diffusion
        img += delta_t * c * lap

    return img
