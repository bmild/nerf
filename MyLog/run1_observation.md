# Run1 Observation

## Data
`image_4/` has higher resolution compared with `images_8/`
- `images` (Original data) has higher resolution compared with `images_4/`
- All folders contain images from 00-19. (Though their name is different)
Train data: [ 1  2  3  4  5  6  7  9 10 11 12 13 14 15 17 18 19]
Test and Val data: [0 8 16]

## Training - `train()` (from line 575)
- `nerf` model is created and define in `create_nerf(args)` function.

- main part are on the `Core optimization loop`.

## Nerf: `create_nerf(args)` (from line 378): initialize the NeRE's MLP model
- `init_nerf_model` (from `run_nerf_helpers.py`)
	


