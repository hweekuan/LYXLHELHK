# display plots etc
import matplotlib.pyplot as plt
import numpy as np
import torch

class myio:

    def __init__(self):
        self.ncol = 6
        self.nrow = 10
        self.max_page = 0
        self.datastr = 'data'
        self.labelstr = 'label'
        return
    # ====================================================
    def make_filename(self,base,batch_idx,page_num):
        filename = base+'b'+str(batch_idx)+'n'+str(page_num)+'.png'
        return filename
    # ====================================================
    def print_to_page(self,plt,batch_idx,page_num):
        filename = self.make_filename('./proj',batch_idx,page_num)
        plt.savefig(filename,bbox_inches='tight') 

    # ====================================================
    #
    # given original image - batch? projected pts - batch?
    # number of dimension used to project this point
    # for example output image name need to append number_dim 
    # print out two image with filename: 
    #    ID_class_number_dim_epoch_original
    #    ID_class_number_dim_epoch_projected
    #
    # ====================================================
    def stream_out(self,cln_pts,pred_list,lab_list,batch_idx):

        cln_pts = cln_pts.detach().cpu().numpy()

        fig = plt.figure(num=None, figsize=(12, 15), dpi=80, facecolor='w', edgecolor='k')
        plt.axis('off')
        img_idx = 0
        page_num = 0

        for cln,pred,lab in zip(cln_pts,pred_list,lab_list):

            # make cln and prj image of the form (height, width, channels)
            #
            cln_img = cln*255
            cln_img = np.rollaxis(cln_img,0,3).astype(np.uint8)

            # squeeze dimensions for MNIST (28,28,1) -> (28,28)
            cln_img = np.squeeze(cln_img)

            if img_idx < self.nrow*self.ncol:
                pre_str = str(pred.detach().cpu().numpy())
                lab_str = str(lab.detach().cpu().numpy())
                plt.subplot(self.nrow, self.ncol, img_idx+1)
                plt.title('lab'+str(lab_str))
                showfig1 = plt.imshow(cln_img)
                showfig1.axes.get_xaxis().set_visible(False)
                showfig1.axes.get_yaxis().set_visible(False)
                plt.subplot(self.nrow, self.ncol, img_idx+2)
                plt.title('pre'+str(pre_str))
                showfig2 = plt.imshow(cln_img)
                showfig2.axes.get_xaxis().set_visible(False)
                showfig2.axes.get_yaxis().set_visible(False)
                img_idx += 2
            else:
                self.print_to_page(plt,batch_idx,page_num)
                plt.clf()
                break


        plt.close(fig)


