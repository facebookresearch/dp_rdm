# how many embeddings? 
import numpy as np
from tqdm import tqdm
import os
from shutterstock_utils import SHUTTERSTOCK_NPZ_METACLIP_BASEPATH, NUM_FILES

def parse_fname(fname):
    fname_split = fname.split('/')
    return [int(i) for i in fname_split[-2].split('_')] + [int(fname_split[-1].split('.')[0])]

if __name__ == "__main__":    
    files = []
    for folder in sorted(os.listdir(SHUTTERSTOCK_NPZ_METACLIP_BASEPATH)):
        for npzfile in sorted(os.listdir(f'{SHUTTERSTOCK_NPZ_METACLIP_BASEPATH}/{folder}')):
            files.append(f'{SHUTTERSTOCK_NPZ_METACLIP_BASEPATH}/{folder}/{npzfile}')
    FILE_PARTS = 5
    index = np.random.randint(0, 10, size=(NUM_FILES, FILE_PARTS))
    print(index.shape)
    count = 0
    
    start_rec = 0
    for file_path in (pbar := tqdm(files)):
        halt = False
        n_samples = np.load(file_path)['arr_0'].shape[0]
        if count + n_samples > NUM_FILES:
            n_samples = NUM_FILES - count
            halt = True
        
        
        count += n_samples
        file_quad = parse_fname(file_path)

        # Add block:
        A = np.zeros((n_samples, FILE_PARTS))
        A[:, 0:4] = file_quad
        A[:, -1] = np.arange(n_samples)

        index[start_rec:start_rec+n_samples] = A
        
        start_rec = start_rec + n_samples 
        pbar.set_description(f"Processed... {count} \t {index[count - 1]}")
        if halt:
            break
        

    np.savez_compressed(f'shutterstock_files_index_{NUM_FILES}.npz', index)