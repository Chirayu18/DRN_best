from PIL import Image

# Take list of paths for images
nepochs = 99
dirname = "ppw/"
image_path_list = []
for i in range(nepochs):
    image_path_list.append(dirname+"ep_"+str(i+1)+"_agg_0.png")
print(image_path_list)

# Create a list of image objects
image_list = [Image.open(file) for file in image_path_list]

# Save the first image as a GIF file
image_list[0].save(
            'animation.gif',
            save_all=True,
            append_images=image_list[1:], # append rest of the images
            duration=100, # in milliseconds
            loop=0)
