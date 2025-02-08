#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def otsu_threshold_from_array(data, nbins=256):
    """
    Compute Otsu's threshold for a one-dimensional array of numbers.

    Parameters:
        data (np.ndarray): 1D array of data points.
        nbins (int): Number of bins to use for the histogram.

    Returns:
        threshold (float): The computed threshold value.
        hist (np.ndarray): The histogram counts.
        bin_centers (np.ndarray): The centers of the histogram bins.
    """
    # Compute the histogram of the data.
    hist, bin_edges = np.histogram(data, bins=nbins)
    # Calculate bin centers from the edges.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = data.size

    # Calculate the total weighted sum (sum of bin_center * count)
    sum_total = np.dot(hist, bin_centers)

    sumB = 0.0           # Cumulative sum for the background
    weightB = 0.0        # Cumulative weight for the background
    max_between_variance = 0.0  # Maximum between-class variance found so far
    threshold = bin_centers[0]  # Initialize threshold to first bin center

    # Iterate over all bins to test each possible threshold
    for i in range(nbins):
        weightB += hist[i]  # Background weight (number of samples in background)
        if weightB == 0:
            continue  # Skip if no data in the background yet

        weightF = total - weightB  # Foreground weight (remaining samples)
        if weightF == 0:
            break  # Stop if no data remains for the foreground

        # Update cumulative sum for background.
        sumB += bin_centers[i] * hist[i]
        meanB = sumB / weightB           # Mean of the background
        meanF = (sum_total - sumB) / weightF  # Mean of the foreground

        # Calculate the between-class variance
        between_variance = weightB * weightF * (meanB - meanF) ** 2

        # If a new maximum variance is found, update the threshold.
        if between_variance > max_between_variance:
            max_between_variance = between_variance
            threshold = bin_centers[i]

    return threshold, hist, bin_centers


def KDE_valley(data):
    kde = gaussian_kde(np.log10(data))
    x_values = np.linspace(np.log10(UMI).min() - 1, np.log10(UMI).max() + 1, 1000)
    kde_values = kde(x_values)
    peaks, _ = find_peaks(kde_values)

    # Find valleys by looking for peaks in the negative of the KDE values (i.e. local minima of the original KDE).
    valley_indices, _ = find_peaks(-kde_values)

    # We assume that the data has two main peaks:
    if len(peaks) >= 2 and len(valley_indices) > 0:
        # Sort peaks in ascending order of their positions.
        left_peak = peaks[0]
        right_peak = peaks[-1]
    
    # Among the detected valleys, choose those that lie between the two peaks.
        candidate_valleys = valley_indices[(valley_indices > left_peak) & (valley_indices < right_peak)]
    
        if candidate_valleys.size > 0:
        # Select the valley with the lowest density (i.e. minimum KDE value) among candidates.
            valley_index = candidate_valleys[np.argmin(kde_values[candidate_valleys])]
        else:
        # Fallback: if no valley is found between peaks, take the valley with the lowest KDE value.
            valley_index = valley_indices[np.argmin(kde_values[valley_indices])]
    else:
        valley_index = None

    # --- Output and Visualization ---
    if valley_index is not None:
        threshold = x_values[valley_index]
        return threshold, x_values, kde_values
    else:
        return np.nan, x_values, kde_values

def sturges_formula (data):
    """
    Calculate the optimal number of histogram bins and bin width using the sturges_formula.
    
    Parameters:
        data (np.array): 1-D array of data points.
        
    Returns:
        nbins (int): Optimal number of bins.
        bin_width (float): The width of each bin.
    """
    n = len(data)
    nbins = int(np.ceil(np.log2(n) + 1))
    return nbins

def square_root_choice (data):
    n = len(data)
    nbins = int(np.ceil(np.sqrt(n)))
    return nbins

def freedman_diaconis_rule(data):
    """
    Calculate the optimal number of histogram bins and bin width using the Freedman-Diaconis rule.
    
    Parameters:
        data (np.array): 1-D array of data points.
        
    Returns:
        nbins (int): Optimal number of bins.
        bin_width (float): The width of each bin.
    """
    data = np.asarray(data)
    n = data.size

    # Compute the 25th and 75th percentiles to get the Interquartile Range (IQR)
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25

    # Calculate the bin width using Freedman-Diaconis rule.
    # If the IQR is zero (e.g., if data are nearly identical), use a fallback.
    bin_width = 2 * iqr / np.cbrt(n)
    if bin_width <= 0:
        bin_width = 1  # Fallback to a default bin width

    # Calculate the number of bins by dividing the data range by the bin width.
    data_range = data.max() - data.min()
    nbins = int(np.ceil(data_range / bin_width))
    
    return nbins, bin_width

if __name__ == '__main__':
    np.random.seed(0)

    # Create sample data for a half noise distribution and a signal distribution.
    # Noise: Only the positive half of a normal distribution (half-normal)
    noise = np.abs(np.random.normal(loc=0, scale=1, size=1000))
    
    # Signal: A full normal distribution centered at 5
    signal = np.random.normal(loc=5, scale=1, size=1000)
    
    # Combine the data
    data = np.concatenate([noise, signal])

    # Compute Otsu's threshold on the array data.
    threshold, hist, bin_centers = otsu_threshold_from_array(data, nbins=256)
    print("Otsu's threshold:", threshold)

    # Plot the histogram of the data and mark the threshold.
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist, width=(bin_centers[1] - bin_centers[0]),
            color='gray', alpha=0.7, label='Histogram')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f"Threshold = {threshold:.2f}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title("Otsu's Method on Data with a Half Noise Distribution")
    plt.legend()
    plt.show()