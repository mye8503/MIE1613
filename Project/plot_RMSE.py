import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

if __name__ == "__main__":

    models = {
        # 3 month, 6 month, 1 year
        'GBM':    {'less_volatile': [168.92349649120743, 200.21720701474027, 274.9527880451309], 
                'volatile': [45.00583024507942, 63.535267332584695, 135.48526214764263], 
                'combined': [16.61850804094654, 22.762255418852632, 33.114616657868034]},
        'Heston': {'less_volatile': [134.46635396694668, 210.84807472987683, 277.3973605867635], 
                'volatile': [54.268115484574366, 70.65809084881431, 129.06709943853502], 
                'combined': [18.409990253666148, 22.607768157503724, 33.93087570099093]},
        'Merton': {'less_volatile': [153.03202579405084, 128.07179240570312, 166.4664547461578], 
                'volatile': [32.726058735223226, 40.186233832378353, 72.39789432982815], 
                'combined': [11.417739873543944, 16.735929951196105, 22.06522927304602]},
        'SVJ':    {'less_volatile': [171.55985493654848, 232.346079052027, 324.25359855204283], 
                'volatile': [53.283818342254484, 69.95401976549691, 129.8740992984672], 
                'combined': [16.2555437899381, 22.79903307237776, 33.40493971422062]},
    }

    colors = {
        'GBM':    'steelblue',
        'Heston': 'orange',
        'Merton': 'green',
        'SVJ':    'red',
    }

    markers = {
        'GBM':    'o',
        'Heston': 's',
        'Merton': 'D',
        'SVJ':    '^',
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    legend_names = []
    for model, data in models.items():
        color  = colors[model]
        marker = markers[model]

        temp, = ax.plot([0,1,2], data['less_volatile'], color=color, marker=marker,
                linestyle='-',  linewidth=1.5, markersize=6, label=f"{model} less_volatile")

        temp1, = ax.plot([0,1,2], data['volatile'], color=color, marker=marker,
                linestyle='--', linewidth=1.5, markersize=6, label=f"{model} volatile")
        
        temp2, = ax.plot([0,1,2], data['combined'], color=color, marker=marker,
                linestyle='-.', linewidth=1.5, markersize=6, label=f"{model} combined")


    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['3M', '6M', '1Y'])
    ax.set_xlabel('Time Horizon')
    ax.set_ylabel('Mean RMSE')
    ax.set_title('Comparison of Mean RMSE Across Models')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # # --- Legend ---
    # # Line-style legend (Less Volatile / Volatile)
    # style_legend = [
    #     mlines.Line2D([], [], color='black', linestyle='-',  label='Less Volatile'),
    #     mlines.Line2D([], [], color='black', linestyle='--', label='Volatile'),
    # ]

    # # Model/colour legend
    # model_legend = [
    #     mlines.Line2D([], [], color=colors[m], marker=markers[m],
    #                 linestyle='None', label=m)
    #     for m in models
    # ]

    # legend1 = ax.legend(handles=style_legend, loc='upper right', frameon=False)
    # ax.add_artist(legend1)
    # ax.legend(handles=model_legend, loc='center right', frameon=False)

    plt.tight_layout()
    plt.savefig('rmse_comparison.png', dpi=150)
    plt.show()