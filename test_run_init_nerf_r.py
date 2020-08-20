from run_nerf_helpers import *

encoder, decoder = init_nerf_r_models(
        D=8, W=256, input_ch_image= (200,200,3),
        input_ch_coord=60, output_ch=3, skips=[4],
        input_ch_views=30, use_viewdirs=True)

print(encoder.summary())

# print(model.trainable_variables)