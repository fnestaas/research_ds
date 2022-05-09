import matplotlib.pyplot as plt 
import joblib 
import numpy as np 
"""
folders = [
    'odefunc_auto_backward',
    'odefunc_manual_backward',
    'pdefunc_auto_backward',
    'pdefunc_manual_backward',
]"""

folders = [ 
    'PDEFuncautodiffskew', 
    'PDEFuncmanual',
    'PDEFuncmanualskew',
]

fig, axs = plt.subplots(2, 2)
y_max = 0

for folder in folders:
    adjoints = joblib.load(folder + '/adjoint_norm.pkl')
    if len(adjoints) > 0:
        mean_var = [np.mean(np.var(adjoint, axis=-1)) for adjoint in adjoints]
        epochs = np.arange(len(adjoints))
        y_max = max([y_max, 1.1*max(mean_var)])
        axs[0, 0].plot(
            epochs,
            mean_var, 
            label=folder
        )
        sample = int(.75*len(adjoints))
        sample_adjoint = adjoints[sample][0, :]
        axs[0, 1].plot(
            np.arange(len(sample_adjoint)), 
            sample_adjoint,
            label=folder
        )

        nfes = joblib.load(folder + '/num_steps.pkl')
        mean_nfe = [np.mean(nfe) for nfe in nfes]
        if len(mean_nfe) > 0:
            axs[1, 0].plot( 
                epochs, 
                mean_nfe, 
                label=folder 
            )

            state_norm = joblib.load(folder + '/state_norm.pkl')
            mean_state_norm_var = [np.mean(np.var(adjoint, axis=-1)) for adjoint in state_norm]
            axs[1, 1].plot( 
                epochs, 
                mean_state_norm_var, 
                label=folder
            )

axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].legend()
axs[1, 1].legend()

axs[0, 0].set_title('variance of adjoint norm')
axs[0, 1].set_title('sample adjoint')
axs[1, 0].set_title('number of function evaluations')
axs[1, 1].set_title('variance of state norm')
    
plt.show()
