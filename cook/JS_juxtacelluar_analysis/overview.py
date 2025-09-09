import warnings

from kitchen.loader.ephys_loader import cohort_loader
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.video import custom_extraction, format_converter

warnings.filterwarnings("ignore")

def pre_conversion():
    hft_data_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\PassivePuff_JuxtaCelluar_FromJS\SST_EXAMPLE\video"
    format_converter.stack_tiff_to_video(hft_data_path)
   

if __name__ == "__main__":
    # pre_conversion()
    data_set = cohort_loader("PassivePuff_JuxtaCelluar_FromJS", "SST_EXAMPLE")
    special_cellsession = data_set.select("cellsession", coordinate=lambda coordinate: coordinate.session_id == "241122_004_165211").nodes[0]
    print(special_cellsession)

    special_trials = data_set.subtree(special_cellsession).select("trial")
    trials_synced = sync_nodes(special_trials, ("VerticalPuffOn",))

    
    import matplotlib.pyplot as plt
    
    # Set up a beautiful, larger figure
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300, constrained_layout=True)
    
    # Define beautiful colors
    vm_color = "#000000"  # Dark blue-gray for membrane potential
    whisker_color = '#27AE60'  # Green for whisker
    early_spike_color = '#E74C3C'  # Red for early spikes
    late_spike_color = "#3F34DB"  # Blue for late spikes
    other_spike_color = '#95A5A6'  # Gray for other spikes
    stim_color = "#CACACA"  # Light blue for stimulation period
    
    # Flags to track legend entries
    labels_added = {
        'vm': False,
        'whisker': False,
        'early_spikes': False,
        'late_spikes': False,
        'other_spikes': False
    }
    
    for trial_node in trials_synced:
        y_offset = 1 * trial_node.coordinate.temporal_uid.chunk_id
        spiking_component = trial_node.data.potential.hp_component(300)
        spike_indices = trial_node.data.potential.spikes.t
        
        # Plot membrane potential with better styling
        vm_label = 'Vm' if not labels_added['vm'] else ''
        ax.plot(spiking_component.t, spiking_component.v + y_offset,
                lw=0.8, alpha=0.8, color=vm_color, label=vm_label)
        if vm_label:
            labels_added['vm'] = True
        
        # Plot whisker deflection with better styling
        whisker_label = 'Whisker' if not labels_added['whisker'] else ''
        ax.fill_between(trial_node.data.whisker.t, trial_node.data.whisker.v * 0.5 + y_offset, y_offset,
                        lw=0, alpha=0.6, color=whisker_color, zorder=-5, label=whisker_label)
        if whisker_label:
            labels_added['whisker'] = True
        
        # Plot spikes with enhanced visibility
        for spike_id, spike_time in enumerate(spike_indices):
            very_spike = spiking_component.segment(spike_time - 0.005, spike_time + 0.005)
            if 0 <= spike_time <= 0.03:
                early_label = 'Early spikes' if not labels_added['early_spikes'] else ''
                ax.plot(very_spike.t, very_spike.v + y_offset, 
                       color=early_spike_color, lw=1.2, alpha=0.9, label=early_label)
                if early_label:
                    labels_added['early_spikes'] = True
            elif 0.8 >= spike_time >= 0.02:
                late_label = 'Late spikes' if not labels_added['late_spikes'] else ''
                ax.plot(very_spike.t, very_spike.v + y_offset, 
                       color=late_spike_color, lw=1.0, alpha=0.8, label=late_label)
                if late_label:
                    labels_added['late_spikes'] = True
            else:
                other_label = 'Other spikes' if not labels_added['other_spikes'] else ''
                ax.plot(very_spike.t, very_spike.v + y_offset, 
                       color=other_spike_color, lw=0.8, alpha=0.6, label=other_label)
                if other_label:
                    labels_added['other_spikes'] = True
    
    # Highlight stimulation period
    ax.axvspan(0, 0.5, color=stim_color, alpha=0.5, zorder=-10, lw=0,
               label='Airpuff 500ms')
    
    # Enhanced styling
    ax.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
    ax.set_ylabel("# Trial", fontsize=14, fontweight='bold')
    ax.set_title("Juxta & Whisking", 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(-2, 4)
    ax.set_yticks([0, 5, 10, 15, 20])
    
    # Add legend
    ax.legend(loc='best', framealpha=0.9, fontsize=11, ncol=2)
    
    # # Improve tick styling
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # Add subtle background
    # ax.set_facecolor('#FAFAFA')
    
    # plt.savefig('tmp.png', dpi=300)
    plt.show()





    # custom_extraction.default_collection(data_set, format=".mp4")
