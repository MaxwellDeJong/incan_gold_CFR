import numpy as np
import torch
import pickle

class MasterBuffer():

    def __init__(self, size):

        self.memory = [None] * size
        self.size = size

        self.n_entries = 0
        self.n_entries_added = 0


    def add(self, local_buffer):

        for i in range(len(local_buffer)):

            if self.n_entries < self.size:
                self.memory[self.n_entries] = local_buffer.memory[i]
                self.n_entries += 1
                self.n_entries_added += 1

            else:

                self.n_entries_added += 1

                acceptance_prob = self.n_entries_added / self.size

                if np.random.random() < acceptance_prob:
                    idx = np.random.randint(0, self.size)
                    self.memory[idx] = local_buffer.memory[i]

        local_buffer.reset()


    def __len__(self):
        return self.n_entries


    def recast(self, n_samples):

        n_samples = min(n_samples, self.n_entries)

        sample_idx = np.random.randint(0, self.n_entries, n_samples)

        input_size = len(self.memory[0][0])
        action_size = len(self.memory[0][2])

        x_train = np.zeros((n_samples, input_size))
        t_arr = np.zeros(n_samples)
        action_train = np.zeros((n_samples, action_size))

        for i in range(n_samples):
            x_train[i, :] = self.memory[sample_idx[i]][0]
            t_arr[i] = self.memory[sample_idx[i]][1]
            action_train[i, :] = self.memory[sample_idx[i]][2]

        return (torch.from_numpy(x_train).float(), torch.from_numpy(t_arr).float(), torch.from_numpy(action_train).float())


    def save(self, prefix_str, process_id, initial_num):

        filename = '/media/max/NVME/IG/' + prefix_str + str(process_id) + '-'

        for i in range(self.len()):
            x_train = torch.from_numpy(self.memory[i][0])
            t_arr = torch.from_numpy(self.memory[i][1])
            action_train = torch.from_numpy(self.memory[i][2])

            torch.save(x_train, filename + 'x' + '_' + str(initial_num + i))
            torch.save(t_arr, filename + 't' + '_' + str(initial_num + i))
            torch.save(action_train, filename + 'y' + '_' + str(initial_num + i))
            
        self.memory = [None] * self.size

        self.n_entries = 0
        self.n_entries_added = 0
