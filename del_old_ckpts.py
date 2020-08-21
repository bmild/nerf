import os

def del_ckpts(basedir='./logs/', expname=''):    
    if expname == '':
        exps = [exp
                for exp in os.listdir(basedir)
                if os.path.isdir(os.path.join(basedir, exp))]
        exps.remove('summaries')

        # print(exps)
        for exp in exps:
            del_ckpts(basedir,exp)
        return
        

    ckpts_opt = [os.path.join(basedir, expname, f) 
            for f in sorted(os.listdir(os.path.join(basedir, expname))) 
            if 'optimizer' in f and f.endswith('.npy')]
    
    if len(ckpts_opt) == 0:
        # no ckpts to remove
        return
    
    last_cpkt_str = ckpts_opt[-1][-10:-4]

    keeps = [os.path.join(basedir, expname, f) 
        for f in os.listdir(os.path.join(basedir, expname)) 
        if f.endswith(f'{last_cpkt_str}.npy')]

    removes = [os.path.join(basedir, expname, f) 
        for f in os.listdir(os.path.join(basedir, expname)) 
        if f.endswith('.npy') and not f.endswith(f'{last_cpkt_str}.npy')]

    print(f'\nDeleting old ckpts for experiment {expname}')
    print(f'{len(keeps)} ckpts with iteration {last_cpkt_str} will be kept')
    print(f'Other {len(removes)} cpkts will be removed')

    if len(removes) == 0:
        print('Skipping')
        return

    confirm = input('Confirm?\n> ')
    if confirm != 'yes' and confirm != 'y':
        print('Aborting')
        return

    for f in removes:
        os.remove(f)
    
    print('Complete')


if __name__ == '__main__':
    del_ckpts(expname='')





