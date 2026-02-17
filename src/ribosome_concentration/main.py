import typer

import ribosome_concentration.input.input as inp
import ribosome_concentration.EM.compartment_based as comp
import mrcfile
import napari
import os 
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd

app = typer.Typer()

@app.command()
def analyse_dir(dir: str, 
                star_path: str, 
                tomogram_apx: float, 
                star_file_apx: float, 
                results_dir: str, 
                z_separation: int = 40,
                y_separation: int = 40,
                x_separation: int = 40,
                compartment_size: int = 40,
                membrane: bool = False,
                membrane_path: str = None,
                subset: int=None):
    
    tomogram_apx = tomogram_apx
    star_file_apx = star_file_apx
    tomogram_segmented_dir = results_dir

    star = inp.read_star_file(star_path)


    os.makedirs(tomogram_segmented_dir, exist_ok=True)
    os.makedirs(f'{tomogram_segmented_dir}/tomograms/', exist_ok=True)

    tomograms = []
    os.makedirs(tomogram_segmented_dir, exist_ok=True)
    os.makedirs(f'{tomogram_segmented_dir}/tomograms/', exist_ok=True)
    tomogram_segmented_info_df_rows = []
    files_to_examine = os.listdir(dir)
    if subset is not None:
        files_to_examine = files_to_examine[:subset]

    for f in tqdm(files_to_examine):
        try:
            if f.endswith('.mrc'):
                print(f)
                root_name = f.split('_21.00Apx.mrc')[0]
                name = root_name + ".tomostar"
                star_filt = star.loc[star.rlnMicrographName == name, :]
                mat = star_filt[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values.astype(int)

                reconstruction_path = dir + "/deconv/" + f"{root_name}_21.00Apx.mrc"
                if membrane_path is None:
                    membrane_path = "deconv/Predictions/" + f"{root_name}_21.00Apx_semantic.mrc"


                tomogram, _ = inp.read_tomogram(reconstruction_path)
                if membrane:
                    membranes_mask = mrcfile.read(membrane_path)

                tomo = comp.Tomogram(tomogram, mat, tomogram_apx=tomogram_apx, star_file_apx=star_file_apx, name=root_name, save_path=f'{tomogram_segmented_dir}/tomograms/{root_name}')
                os.makedirs(f'{tomo.save_path}', exist_ok=True)
                tomo.compartmentalize(compartment_size=compartment_size, 
                                      x_separation=x_separation, 
                                      y_separation=y_separation, 
                                      z_separation=z_separation)
                tomo.obtain_compartment_stdevs()
                tomo.obtain_compartment_counts()


                tomo.reset_forbidden_compartments()
                tomo.find_void_volume(plot=True, proportion_compartments_threshold=0.6)
                if membrane:
                    tomo.find_forbidden_compartments(membranes_mask, threshold=0.001, label="contains_membrane")
                tomo.forbid_empty_compartments()

                reasons_to_forbid = ['void', 'contains_membrane']
                tomo.calculate_ribosome_concentration(reasons_to_forbid=reasons_to_forbid)
                tomograms.append(tomo)
                if tomo.suspect:
                    print(f"Tomogram {tomo.name} is suspect owing to poor segmentation of the void volume.")


                row_info_df = pd.DataFrame({
                    'name': [tomo.name],
                    'ribosome_concentration_molar' : [tomo.ribosome_concentration_molar],
                    'reasons_to_forbid': [';'.join(reasons_to_forbid)],
                    'total_volume' : [tomo.total_volume],
                    'allowed_volume' : [tomo.allowed_volume_L],
                    'total_num_ribosomes': [tomo.total_num_particles],
                    'allowed_ribosomes': [tomo.number_of_ribosomes_in_allowed_volume],
                    'void_suspect': [tomo.suspect]
                })

                tomogram_segmented_info_df_rows.append(row_info_df)

                with open(f'{tomo.save_path}.pkl', 'wb') as file:
                    pickle.dump(tomo, file)



        except Exception as e:
            print('Processing failed for', f)
            print(e)

    tomogram_segmented_info_df = pd.concat(tomogram_segmented_info_df_rows)

    tomogram_segmented_info_df.to_csv(f'{tomogram_segmented_dir}/tomogram_segmented_info_df.csv')





@app.command()
def view_segmented_tomogram(pickle_path):
    with open(pickle_path, 'rb') as f:
        tomo = pickle.load(f)

    tomo.view_segmented_tomogram()