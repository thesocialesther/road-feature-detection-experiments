# Road dataset snapshots
Two local split folders were found:
- split_dataset2: 4,298 files, 293,935,374 bytes.
- split_dataset3: 5,704 files, 414,075,021 bytes.

The original autonomous_vehicle.yaml points to split_dataset2. Both snapshots have complete filename/size/SHA-256 inventories and their available label/configuration files in datasets/. Image files are not included in this initial source export. Neither snapshot is asserted to be the final 8,998-image dataset.

For training, place the matching images under the respective train/images, val/images and test/images directories, and extract labels.zip into the same dataset root. Set the YAML path to that full root. The image filenames must match the label stems.

Full image distribution and a permanent download link remain pending identification of the final dataset version. Do not combine the two snapshots or describe the annotations-only export as a complete image dataset.
