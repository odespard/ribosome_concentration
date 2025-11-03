import os
import mrcfile

def read_tomogram(path):
    if os.path.exists(path):
        tomogram = mrcfile.read(path)
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Loaded tomogram {name}.")
        return tomogram, name
    else:
        return None, None
    


def read_star_file(path: str) -> pd.DataFrame:
    star = starfile.read(path)

    if type(star) is dict:
        star = star['particles']

    return star



def rescale_star_coordinates(mat, tomogram_apx, star_file_apx):
    rescaled_mat = mat.T // int(tomogram_apx / star_file_apx)
    return rescaled_mat