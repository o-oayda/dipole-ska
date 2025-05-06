# dipole-ska

## Requirements
We are implementing this on `python 3.12`.
To get set up and working:
1. Clone this repo
2. `cd` into it
3. Make a python virtual environment: `python -m venv .venv`
4. Install the requirements: `pip install -r requirements.txt`

We are using `matplotlib==3.7.5` to avoid the plotting issue where the contours wrap around the coordinates.

## Adding the Data
Place the SKA maps in their corresponding folder in `data/ska/`.
For example, the maps in `SKA_Briggs1m_AA` would go in `data/ska/briggs_1/AA/`.

> [!NOTE]
> The `.placeholder` files in `data/ska/` are there to make sure the empty directories are tracked by git.