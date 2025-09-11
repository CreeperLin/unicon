import time
import argparse
import numpy as np

from unicon.states import states_get, states_init

def generate_data(data_ref):
    # Example: generate random data based on the type of data_ref
    if isinstance(data_ref, np.ndarray):
        return np.random.rand(*data_ref.shape).astype(data_ref.dtype)
    elif isinstance(data_ref, (int, float)):
        return type(data_ref)(np.random.rand())
    else:
        raise ValueError(f'Unsupported data type: {type(data_ref)}')
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--key', required=True, help='the key need to be updated')
    parser.add_argument('-itv', '--interval', type=float, default=0.1, help='update intervel (s)')
    args = parser.parse_args()

    states_init(use_shm=True, load=True, reuse=True)

    data_ref = states_get(args.key)

    if data_ref is None:
        print(f'Error: key {args.key} not found in states.')
        return
    
    print(f'Initial value of {args.key}: {data_ref}')

    try:
        while True:
            new_data = generate_data(data_ref)
            data_ref[:] = new_data
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Writing process interrupted...')

if __name__ == '__main__':
    main()