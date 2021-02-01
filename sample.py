from yolo_transfer_learning.marmot_data_utils import create_training_folder, write_yolo_file, write_train_test


# Change the file paths for your use case.
file_in_pos_path = "./datasets/image/"
# file_in_neg_path = "./marmot_dataset/Negative/Raw/"
file_out_path = "./obj/"
xml_in_path = "./datasets/ground_truth/"

# This function searchs all the pictures in the file_in_pos_path and file_in_neg_path folders, create .txt and .xml files 
# for positive images, and save all the images to the target folder file_out_path. 
image_list, text_list, xml_list = create_training_folder(
    file_in_pos_path, xml_in_path, file_out_path
)

# A function to create a txt file that contain class data and bbox information, a specific format of Yolo model.
# It also split dataset into training and testing and return the file names in two different list. 
train_list, test_list, data_stats = write_yolo_file(image_list, text_list, xml_list)

out_path = "./data/"
# A function to write training and testing images names to txt files. It create one train.txt and one test.txt. 
write_train_test(train_list, test_list, out_path)
