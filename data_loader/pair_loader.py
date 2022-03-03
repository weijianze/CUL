import torch
import torch.utils.data as data
import os
import pdb
import os.path
from PIL import Image
import torchvision.transforms as transforms

class PairImageList(data.Dataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 transform=None, 
                 list_reader=None,
                 loader=None,
                 IN_use=True):
        self.root       = root
        self.transform  = transform
        self.IN_use     = IN_use

        if loader is None:
            self.loader = self._img_loader
        if list_reader is None:
            self.list_reader = self._multidata_pair_read
        self.pair_list = self.list_reader(root, list_path)

    def _list_read(self, list_path):
        label_list = []
        path_list =[]
        with open(list_path, 'r') as file:
            for line in file.readlines():
                img_path, label = line.strip().split(' ')
                label_list.append(int(label))
                path_list.append(img_path)
        return path_list, label_list

    def _multidata_pair_read(self, root, fileList):
        pair_list = []
        pair_file_path = 'pair_list.txt'
        if os.path.exists(pair_file_path):
            with open(pair_file_path,'r') as file:
                for line in file.readlines():
                    gallery_path, probe_path = line.strip().split(' ')
                    pair_list.append((gallery_path, probe_path))


        else:
            if len(root) == len(fileList):
                for i in range(len(root)):
                    path_list = []
                    label_list = []
                    with open(fileList[i],'r') as file:
                        for line in file.readlines():
                            imgPath, label = line.strip().split(' ')
                            path_list.append(os.path.join(root[i], imgPath))
                            label_list.append(int(label))
                    num_classes = max(label_list) + 1
                    data_onehot = self._list2onehot(label_list,num_classes)
                    pair_index = (data_onehot.mm(data_onehot.t())-torch.eye(data_onehot.size(0))).nonzero()

                    for gallery_idx,probe_idx in pair_index:
                        if label_list[gallery_idx] == label_list[probe_idx]:
                            pair_list.append((path_list[gallery_idx], path_list[probe_idx]))
                    print('%s is loaded' % (root[i]))
                    
            with open(pair_file_path,"w") as file:
                for gallery_path, probe_path in pair_list:
                    file.write(gallery_path + ' ' + probe_path + '\n')
        return pair_list

    def _list2onehot(self, labels_list, num_classes):
        labels = torch.tensor(labels_list).unsqueeze(1)
        return torch.zeros(labels.size(0),num_classes).scatter_(1,labels,1)   
    '''
    def _multi_list_read(self, root, fileList):
        label_list = []
        path_list =[]
        clsList = []
        if len(root)==len(fileList):        
            for i in range(len(root)):
                cls_max = 0
                with open(fileList[i], 'r') as file:
                    for line in file.readlines():
                        imgPath, label = line.strip().split(' ')
                        label = int(label)
                        # pdb.set_trace()
                        if label>cls_max:
                            cls_max = label
                        path_list.append((root[i], imgPath, sum(clsList)+label))
                        label_list.append(label)
                clsList.append(cls_max+1)
                    
        return path_list, label_list
    
     

    def _pairs_generate(self, root, list_path):
        pair_list = []
        # gallery_path, gallery_labels = self._list_read(gallery_list_path)
        # probe_path, probe_labels = self._list_read()

        paths, labels = self._multi_list_read(root, list_path)

        num_classes = max(labels)+1
        data_onehot = self._list2onehot(labels,num_classes)

        pair_index = (data_onehot.mm(data_onehot.t())-torch.eye(data_onehot.size(0))).nonzero()

        for gallery_idx,probe_idx in pair_index:
            if labels[gallery_idx] == labels[probe_idx]:
                pair_list.append((paths[gallery_idx], paths[probe_idx], labels[gallery_idx]))
        return pair_list
    '''
    def _img_loader(self, path):
        img = Image.open(path).convert('L')
        return img

    def _img_Gaussain_norm(self, img):
        img_mean = img.mean()
        img_std = img.std()+1e-8
        img = (img-img_mean)/img_std
        return img

    def __getitem__(self, index):
        output = {}
        gallery_path, probe_path = self.pair_list[index]
        gallery_img = self._img_loader(gallery_path)
        probe_img = self._img_loader(probe_path)

        if False:
            print(os.path.join(gallery_path))
            print(os.path.join( probe_path))
            print('==================================')

        if self.transform is not None:
            gallery_img = self.transform(gallery_img)
            probe_img = self.transform(probe_img)
        if self.IN_use:
            gallery_img,probe_img= self._img_Gaussain_norm(gallery_img), self._img_Gaussain_norm(probe_img)

        output['data_1'] = gallery_img
        output['data_2'] = probe_img

        return output

    def __len__(self):
        return len(self.pair_list)


if __name__=="__main__":
    root_path = '/home/jianze.wei/DemoCode/'
    gallery_list = 'test_NIR_list.txt'
    probe_list = 'test_VIS_list.txt'
    train_loader = torch.utils.data.DataLoader(
        PairImageList(root=root_path, gallery_list_path=gallery_list, probe_list_path=probe_list, 
        transform=transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()])),
        batch_size=5, shuffle=True,
        num_workers=0, pin_memory=True)
    # pdb.set_trace()
    for epoch in range(2):            
        for batch_idx,data in enumerate(train_loader):
            print(data['data_1'].size())
            print(data['data_2'].size())
            # print(data['label'].size())
            print(data['label'])
