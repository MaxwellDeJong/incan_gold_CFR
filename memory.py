import numpy as np
import torch

class Buffer():

    def __init__(self, size):

        self.memory = [None] * size
        self.size = size

        self.n_entries = 0
        self.n_entries_added = 0


    def form_entry(self, embedded_history, t, arr):

        history_embedding = np.array(list(embedded_history.values()), dtype=np.int)

        return [history_embedding, t, arr]


    def add(self, embedded_history, t, arr):

        entry = self.form_entry(embedded_history, t, arr)
        #print('formed entry: ', entry)

        if self.n_entries < self.size:

            self.memory[self.n_entries] = entry
            self.n_entries += 1
            self.n_entries_added += 1

        else:

            self.n_entries_added += 1

            acceptance_prob = self.n_entries_added / self.size

            if np.random.random() < acceptance_prob:
                idx = np.random.randint(0, self.size)
                self.memory[idx] = entry


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


    def reset(self):

        self.memory = [None] * self.size

        self.n_entries = 0
        self.n_entries_added = 0


    def save(self, loc, pid, initial_num):

        for i in range(self.n_entries):
            filename_prefix = loc + '_' + str(pid) + '_' + str(initial_num + i) + '.pt'

            x = 

            torch.save
