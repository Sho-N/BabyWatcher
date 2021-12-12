import argparse
import os
import shutil
import xml.etree.ElementTree as ET
import glob
from sklearn.model_selection import train_test_split


def copy_image_files(src_dir, dst_dir, target):
    """Copy image files"""
    target_dir = f'{src_dir}/{target}/JPEGImages/*'
    files = glob.glob(target_dir)

    new_files = []
    for file in files:
        new_file = f'{target}_{os.path.basename(file)}'
        shutil.copy2(file, f'{dst_dir}/JPEGImages/{new_file}')
        new_files.append(os.path.splitext(new_file)[0])

    return new_files


def modify_xml_files(src_dir, dst_dir, target):
    """Modify XML files"""
    target_dir = f'{src_dir}/{target}/Annotations/*'
    files = glob.glob(target_dir)

    new_files = []
    for file in files:
        xml = open(file)
        tree = ET.parse(file)
        root = tree.getroot()

        # Rewrite <filename>
        for f in root.iter('filename'):
            new_file = f'{target}_{os.path.basename(file)}'
            f.text = new_file
            tree.write(f'{dst_dir}/Annotations/{new_file}', encoding='UTF-8')
            new_files.append(os.path.splitext(new_file)[0])

    return new_files


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='data/multiple_dataset/', help='path/to/src_dir/')
    parser.add_argument('--dst_dir', type=str, default='data/merged_dataset/', help='path/to/dst_dir/')
    opt = parser.parse_args()
    return opt


def main(opt):
    # set work dir
    src_dir = opt.src_dir
    dst_dir = opt.dst_dir

    # get dataset list
    _listdir = os.listdir(src_dir)
    targets = [f for f in _listdir if os.path.isdir(os.path.join(src_dir, f))]

    # get label
    f = open(f'{src_dir}/{targets[0]}/labelmap.txt')
    _read = f.read().split()
    f.close()
    labels = [i.split(":")[0] for i in _read]
    labels = labels[2:]

    # Create the destination folder
    # Remove
    if os.path.exists(dst_dir):
        print('Remove existed folder...')
        shutil.rmtree(dst_dir)
    # Create
    os.makedirs(dst_dir)
    os.makedirs(f'{dst_dir}/Annotations')
    os.makedirs(f'{dst_dir}/JPEGImages')
    os.makedirs(f'{dst_dir}/ImageSets')
    os.makedirs(f'{dst_dir}/ImageSets/Main')

    # Merge src to dst
    files = []
    for target in targets:
        _files = copy_image_files(src_dir, dst_dir, target)
        _ = modify_xml_files(src_dir, dst_dir, target)
        files.extend(_files)

    # Create labels
    str_ = '\n'.join(labels)
    with open(f'{dst_dir}/labels.txt', 'w', encoding='utf-8') as f:
        f.writelines(str_)

    # Split
    trainval, test = train_test_split(files, test_size=0.1, random_state=0)
    train, val = train_test_split(trainval, test_size=0.1, random_state=0)

    # Create lists
    splits_txt = ['trainval', 'train', 'val', 'test']
    splits_var = [trainval, train, val, test]
    for t, v in zip(splits_txt, splits_var):
        str_ = '\n'.join(v)
        with open(f'{dst_dir}/ImageSets/Main/{t}.txt', 'w', encoding='utf-8') as f:
            f.writelines(str_)


if __name__ == "__main__":
    option = parse_opt()
    main(option)