import numpy as np

class SeparatedDataset():

    def __init__(self, data, num_task):
        # returns the dataset required for all tasks  
        # Random splitting
        # ensuring parent class is properly initialized -> mantaining integrity by super()
        super(SeparatedDataset, self).__init__()
        self.num_tasks = num_task
        self.data = data
        self.task_masks = []                   # Initialize an empty list to store the masks
        self.split_train_mask()
        
    def split_train_mask(self):

        # Convert to numpy array if it's not already
        # if not isinstance(train_mask, np.ndarray):
        #    train_mask = np.array(train_mask)
        
        # get the indices from the mask
        train_indices = np.where(self.data.train_mask)[0]
        
        # shuffle the indices
        np.random.shuffle(train_indices)
        
        # Prepare an array to hold the number of elements in each task
        # len(train_indices) = 10, num_tasks = 2, split_sizes = [5, 5]
        split_sizes =  [len(train_indices) // self.num_tasks] * self.num_tasks

        # handle remaining instances
        for i in range(len(train_indices) % self.num_tasks):
            split_sizes[i] += 1

        # Array to hold the actual indices based on the last element index
        split_indices = np.split(train_indices, np.cumsum(split_sizes)[:-1]) # split train_indices by summing split_indices

        # Initialize and set the corresponding indices to True in each mask
        for indices in split_indices:
            mask = np.zeros_like(self.data.train_mask, dtype=bool)
            mask[indices] = True
            self.task_masks.append(mask)  # Store each mask in the list
                  
    
    def get_sample(self, sample_size):
        # randomly select indexes for testing
        print("Not required")
        

    