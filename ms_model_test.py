import torch
import os, platform, argparse
from torchvision import transforms
from ms_dataloader import MicroS_Dataset, ImglistToTensor
from convlstmnet import ConvLSTMNet
from torchvision.transforms import ToPILImage
import cv2


def check_device():
    """
    This function checks whether to use GPU (or CPU)
    and whether to use multi-GPU (or single-GPU)
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    multi_gpu = True
    multi_gpu = use_cuda and multi_gpu and torch.cuda.device_count() > 1
    num_gpus = (torch.cuda.device_count() if multi_gpu else 1) if use_cuda else 0

    return use_cuda, device, multi_gpu, num_gpus

def generate_images(model, test_dataloader, device):
    output_folder="outputImages"
    model.eval()
    to_pil = ToPILImage()

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        for _, (input_batch, output_batch) in enumerate(test_dataloader):
            input_batch = input_batch.to(device)

            # Assuming your model takes input_frames, future_frames, and output_frames as arguments
            pred = model(input_batch, 
                         input_frames=test_data.n_frames_input, 
                         future_frames=test_data.n_frames_output, 
                         output_frames=test_data.n_frames_output, 
                         teacher_forcing=False)

            # Assuming pred is a tensor of shape (batch_size, num_frames, channels, height, width)
            for batch_index in range(pred.size(0)):
                for frame_index in range(pred.size(1)):
                    # Get a single frame from the predicted tensor
                    single_frame = pred[batch_index, frame_index]

                    # Convert the tensor frame to PIL image
                    pil_image = to_pil(single_frame)

                    # Save the PIL image to the output folder
                    image_path = os.path.join(output_folder, f"predicted_image_{batch_index}_{frame_index}.jpg")
                    pil_image.save(image_path)

    # Create a video from the saved images
    create_video(output_folder)

def create_video(image_folder, video_name="output_video.avi"):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"DIVX"), 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()




# def generate_images(model, test_dataloader, device):
#     model.eval()
#     to_pil = ToPILImage()

#     with torch.no_grad():
#         for _, (input_batch, output_batch) in enumerate(test_dataloader):
#             input_batch = input_batch.to(device)

#             # Assuming your model takes input_frames, future_frames, and output_frames as arguments
#             pred = model(input_batch, 
#                          input_frames=test_data.n_frames_input, 
#                          future_frames=test_data.n_frames_output, 
#                          output_frames=test_data.n_frames_output, 
#                          teacher_forcing=False)

#             # Assuming pred is a tensor of shape (batch_size, num_frames, channels, height, width)
#             for batch_index in range(pred.size(0)):
#                 for frame_index in range(pred.size(1)):
#                     # Get a single frame from the predicted tensor
#                     single_frame = pred[batch_index, frame_index]

#                     # Convert the tensor frame to PIL image
#                     pil_image = to_pil(single_frame)

#                     # You can now save, display, or process the PIL image as needed
#                     pil_image.save(f"predicted_image_{batch_index}_{frame_index}.jpg")
#                     # Display or process the PIL image as needed
#                     pil_image.show()


# def generate_images(model, test_dataloader, device):
#     model.eval()
#     with torch.no_grad():
#         for _, (input_batch, output_batch) in enumerate(test_dataloader):
#             input_batch = input_batch.to(device)


#             # Assuming your model takes input_frames, future_frames, and output_frames as arguments
#             pred = model(input_batch, 
#                          input_frames=test_data.n_frames_input, 
#                          future_frames=test_data.n_frames_output, 
#                          output_frames=test_data.n_frames_output, 
#                          teacher_forcing=False)

#             # Here you can handle the generated predictions as needed
#             # For example, save the predicted images, display them, etc.
#             # Note: Modify this part based on your specific requirements and model output
#             # For demonstration, let's print the shape of the predicted tensor
#             print("Predicted shape:", pred.shape)

if __name__ == "__main__":
    if platform.system() == "Windows":
        videos_root = os.path.join(os.getcwd(), 'data/test')  
        train_path = os.path.join(videos_root, 'config_file_train') 
        valid_path = os.path.join(videos_root, 'config_file_test') 
        dir_checkpoint = os.path.join(os.getcwd(), '/checkpoints') 
    else:        
        videos_root = os.path.join(os.getcwd(), 'data/test')  
        train_path = os.path.join(os.getcwd(), 'data/test/config_file_train')
        valid_path = os.path.join(os.getcwd(), 'data/test/config_file_test') 
        dir_checkpoint = os.path.join(os.getcwd(), '/checkpoints')   

    use_cuda, device, multi_gpu, num_gpus = check_device()
    print("Use of cuda, num gpus ", num_gpus)

    if use_cuda:
        tot_frames, batch_size, num_epochs, learning_rate = 20, 16, 250, 1e-3
    else:
        tot_frames, batch_size, num_epochs, learning_rate = 8, 1, 5, 1e-3
    # Load the trained model checkpoint
    checkpoint_path = "checkpoint_epoch4.pth"
    model = ConvLSTMNet(input_channels=3, layers_per_block=(3, 3, 3, 3), hidden_channels=(32, 48, 48, 32),
                        skip_stride=2, cell='convlstm', cell_params={"order": 3, "steps": 3, "rank": 8},
                        kernel_size=3, bias=True, output_sigmoid=False)
    
    n_input, n_output = tot_frames//2, tot_frames//2

    checkpoint = torch.load(checkpoint_path)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set the device

    model.to(device)
    transform = transforms.Compose([
        ImglistToTensor()  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    ])

    # Define test dataset and dataloader
    test_data = MicroS_Dataset(
        root=videos_root,
        config_path=valid_path,
        n_frames_input=n_input,
        n_frames_output=n_output,
        imagefile_template='img_{:d}.jpg',
        transform=transform,
        is_train=True
    )

    # test_data = MicroS_Dataset(
    #     root="data/test/dataset",
    #     config_path="data/test/config_file",
    #     n_frames_input=1,  # Specify the number of input frames
    #     n_frames_output=1,  # Specify the number of output frames
    #     imagefile_template='img_{:d}.jpg',
    #     transform=transforms.Compose([ImglistToTensor()]),
    #     is_train=False
    # )
    print("Dataset size:", len(test_data))

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 * max(num_gpus, 1),
        #pin_memory=True
    )

    # test_dataloader = torch.utils.data.DataLoader(
    #     dataset=test_data,
    #     batch_size=20,
    #     shuffle=False,
    #     num_workers=1,
    #     # pin_memory=True
    # )
    print('test_dataloader', test_dataloader)

    # Generate images using the trained model
    generate_images(model, test_dataloader, device)
