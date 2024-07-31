import matplotlib.pyplot as plt

def plot_histograms(*lists):
    """
    Plots a histogram for each list provided as input.
    
    Parameters:
    *lists: A variable number of list objects.
    """
    for i, lst in enumerate(lists, start=1):
        plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
        plt.hist(lst, bins=30, alpha=0.7)  # Adjust bins and alpha as needed
        plt.title(f'Histogram {i}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()  # Display the histogram

# # Example usage:
# list1 = [1, 2, 2, 3, 4, 5, 5, 5, 6]
# list2 = [1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9]
# plot_histograms(list1, list2)