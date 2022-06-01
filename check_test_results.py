from cmath import nan
import matplotlib.pyplot as plt 
import joblib 
import numpy as np 
import models.NeuralODEClassifier as node
import jax.random as jrandom
import jax.numpy as jnp
import matplotlib

# folders = [ 
#     'PDEFuncidentity_skew', 
#     'PDEFuncswish_skew',
#     'PDEFuncsigmoid_skew', 
#     'PDEFuncabs_skew'
# ]

folders = [
    # 'pollution_PDEFunc_skew=True15',
    # 'pollution_RegularFunc_skew=True15',
    # 'pollution_PDEFunc_skew=False14',
    # "pollution_FUNC='PDEFunc'_SKEW=False14",
    # 'pollution_RegularFunc_skew=True1',
    # 'pollution_PDEFunc_skew=False1',
    # 'pollution_PDEFunc_skew=True1',
    'pollution_RegularFunc_skew=True',
    # 'pollution_PDEFunc_skew=False',
    'pollution_PDEFunc_skew=True',
    # "cancer_FUNC='RegularFunc'_SKEW=True0"
]

save = False
uncertainty = True

lower_q = .21
upper_q = 1-lower_q

if not uncertainty:
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    fig, axs = plt.subplots(2, 3, figsize=(10, 15))
    y_max = 0
    for i, folder in enumerate(folders):
        if folder[-2] == 'c':
            lib = 'test/'
            
        else:
            lib = 'tests/'
        adjoints = joblib.load(lib + folder + '/adjoint_norm.pkl')
        
        label=folder
        if len(adjoints) > 0:
            mean_var = [np.nanmean(np.quantile(adjoint, q=upper_q, axis=-1)/np.quantile(adjoint, q=lower_q, axis=-1)) for adjoint in adjoints]
            # nan_mask = np.isnan(np.array([np.quantile(adjoint, q=.79, axis=-1)/np.quantile(adjoint, q=.21, axis=-1) for adjoint in adjoints]))
            # num_nans = np.sum(nan_mask)
            # print(f'got {num_nans/np.size(nan_mask)*100}% nans\n')
            # mean_var = [np.mean(np.var(adjoint, axis=-1)) for adjoint in adjoints]
            epochs = np.arange(len(adjoints))
            y_max = max([y_max, 1.1*max(mean_var)])
            axs[0, 0].plot(
                epochs,
                mean_var, 
                label=label
            )
            sample = int(len(adjoints)) - 1
            try:
                sample_adjoint = adjoints[sample][0, :]
            except:
                sample_adjoint = adjoints[sample]
            axs[0, 1].plot(
                np.arange(len(sample_adjoint)), 
                sample_adjoint,
                label=label
            )
            acc = joblib.load(lib + folder + '/acc.pkl')
            to_plot = [jnp.mean(ac) for ac in acc]
            axs[0, 2].plot(
                jnp.arange(len(to_plot)), 
                to_plot, 
                label=label
            )

            nfes = joblib.load(lib + folder + '/num_steps.pkl')
            mean_nfe = [np.mean(nfe) for nfe in nfes]
            epochs = np.arange(len(mean_nfe))
            axs[1, 0].plot( 
                epochs, 
                mean_nfe, 
                label=folder 
            )

            state_norm = joblib.load(lib + folder + '/state_norm.pkl')
            mean_state_norm_var = [np.mean(np.var(adjoint, axis=-1)) for adjoint in state_norm]
            axs[1, 1].plot( 
                epochs, 
                mean_state_norm_var, 
                label=folder
            )
            
            sample_state_norm = state_norm[sample][0, :]
            axs[1, 2].plot( 
                np.arange(len(sample_state_norm)),
                sample_state_norm,
                label=folder 
            )

    axs[0, 0].legend()

    axs[0, 0].set_title('adjoint norm score')
    axs[0, 1].set_title('sample adjoint norm')
    axs[0, 2].set_title('validation loss')
    axs[1, 0].set_title('number of function evaluations')
    axs[1, 1].set_title('variance of state norm')
    axs[1, 2].set_title('sample state norm')

    fig.tight_layout()

    plt.show()

else:
    matplotlib.rcParams.update({'font.size': 22})
    def score_data(data, stat):
        if stat == 'adjoint_norm':
            return jnp.quantile(data, upper_q, axis=-1) / jnp.quantile(data, lower_q, axis=-1)
        else:
            return data
    start = 16
    n_seeds = 5
    stop = start + n_seeds 
    step = (stop - start) // n_seeds
    stats = ['adjoint_norm', 'num_steps', 'acc']
    stat_names = {'adjoint_norm': 'Adjoint Norm', 'num_steps': 'NFE', 'acc': 'Loss'}
    sup_folder_names = {
        'pollution_RegularFunc_skew=True': 'Regular', 
        'pollution_PDEFunc_skew=True': 'Skew Symmetric',
        'pollution_PDEFunc_skew=False': 'Unrestricted',
    }
    x_labels = {'adjoint_norm': 'Training Steps', 'num_steps': 'Training Steps', 'acc': 'Validation Steps'}
    results = {key: {keyi: [] for keyi in stats} for key in folders}
    for sup_folder in folders:
        for stat in stats:
            for seed in range(start, stop, step):
                folder = sup_folder + str(seed)
                data = joblib.load(f'tests/{folder}/{stat}.pkl')
                if stat != 'acc':
                    data = [jnp.mean(d, axis=0) for d in data]
                    try:
                        results[sup_folder][stat].append(jnp.stack(data[:216]))
                    except:
                        results[sup_folder][stat].append(jnp.array(data))
                else:
                    results[sup_folder][stat].append(jnp.array(data))
            results[sup_folder][stat] = jnp.stack(results[sup_folder][stat])

    fig, axs = plt.subplots(1, len(stats))
    for stat, ax in zip(stats, axs):
        for sup_folder in folders:
            data = results[sup_folder][stat]
            N = data.shape[1]
            score = score_data(data, stat)
            mean = jnp.mean(score, axis=0)
            std = jnp.std(score, axis=0)
            lower = jnp.quantile(score, q=.2, axis=0)
            upper = jnp.quantile(score, q=.8, axis=0)
            ax.plot(mean, label=sup_folder_names[sup_folder])
            ax.fill_between(
                np.arange(N), 
                lower, 
                upper, 
                alpha=.4
            )
            ax.set_xlabel(x_labels[stat])
            if stat == 'acc':
                ax.legend()
            ax.set_title(stat_names[stat])
    plt.show()


if save:
    save = input('do you want to save? (yes/no)')
    if save == 'yes':
        fig.savefig('outputs/air_pollution_results.pdf')
