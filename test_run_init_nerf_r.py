from run_nerf_helpers import *

model = init_nerf_r_model(
        D=8, W=256, input_ch_image= (500,500,3),
        input_ch_coord=60, output_ch=3, skips=[4],
        input_ch_views=30, use_viewdirs=True)

print(model.summary())

print(model.trainable_variables)