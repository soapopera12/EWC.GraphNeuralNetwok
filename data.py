import numpy as np
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split

class SeparatedDataset():

    # separating data from initialize so that it can work for any dataset in the file
    def __init__(self, train_mask, num_task):
        super(SeparatedDataset, self).__init__()
        self.num_tasks = num_task
        self.train_mask = train_mask
        self.task_masks = []   # initialize empty task_masks
        self.split_masks()
    
    def split_masks(self):
        '''
            Creates required task_masks arrays for num_tasks tasks
        '''
        
        # get the indices from the mask
        train_indices = np.where(self.train_mask)[0]
        
        # technique -> randomly shuffling the indices
        np.random.shuffle(train_indices)
        
        # Prepare an array to hold the number of elements in each task
        # len(train_indices) = 10, num_tasks = 2, split_sizes = [5, 5]
        split_sizes =  [len(train_indices) // self.num_tasks] * self.num_tasks

        # handle remaining instances
        for i in range(len(train_indices) % self.num_tasks):
            split_sizes[i] += 1

        # Array to hold the actual indices based on the last element index
        split_indices = np.split(train_indices, np.cumsum(split_sizes)[:-1])  # split train_indices by summing split_indices
        
        # Initialize and set the corresponding indices to True in each mask
        for indices in split_indices:
            mask = np.zeros_like(self.train_mask, dtype=bool)
            mask[indices] = True
            self.task_masks.append(mask)  # Store each mask in the list

    def train_test_split_ewc(self, test_ratio):
        '''
        train_test_split_ewc creates training and testing masks for different tasks of EWC
        '''
        train_task_masks_arr = []
        test_task_masks_arr = []

        for i in range(self.num_tasks):
            
            indices = np.arange(len(self.task_masks[i]))
            
            # Split indices based on the original train_mask
            full_indices = indices[self.task_masks[i] == 1]

            train_indices, test_indices = train_test_split(full_indices, test_size=test_ratio)
            
            # Initialize new masks with zeros
            new_train_mask = np.zeros_like(self.task_masks[i])
            new_test_mask = np.zeros_like(self.task_masks[i])

            # Set appropriate indices to 1
            new_train_mask[train_indices] = 1
            new_test_mask[test_indices] = 1

            train_task_masks_arr.append(new_train_mask)
            test_task_masks_arr.append(new_test_mask)
        
        return train_task_masks_arr, test_task_masks_arr
        
    def preparingLoaders(self, data, batch_size, neighbors_arr, train_task_masks_arr, test_task_masks_arr):
        '''
        preparing training loaders for EWC tasks and one final testing masks 
        '''

        train_loader_arr = []
        test_loader_arr = []
        
        for i in range(self.num_tasks):
            # get train and test splits
            train_loader_arr.append(self.getGraphDataLoader(data, train_task_masks_arr[i], batch_size, neighbors_arr))
            test_loader_arr.append(self.getGraphDataLoader(data, test_task_masks_arr[i], batch_size, neighbors_arr))   
        
        '''
        # combine the task_masks first
        final_test_mask = np.zeros_like(self.train_mask)
        for i in range(len(test_task_masks_arr)):
            final_test_mask |= test_task_masks_arr[i]
        '''
        
        # test_loader = self.getGraphDataLoader(data, final_test_mask, batch_size, neighbors_arr)
            
        return train_loader_arr, test_loader_arr
        
    def getGraphDataLoader(self, data_train, task_masks, batch_size, neighbors_arr):
        '''
        This function takes in a graph data object the required batch_size 
        and neighbors_arr and returns a loader
        '''
        loader=NeighborLoader(
            data_train,    # taking in data_train
            input_nodes=task_masks,
            num_neighbors=neighbors_arr,  
            batch_size=batch_size,
            replace=False,
            shuffle=True
        )
        return loader