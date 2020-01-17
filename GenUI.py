import GenBackend


class UI:

    def __init__(self, d_lr, d_batch, d_epoch, d_keep, early):
        self.dlr = d_lr
        self.dbatch = d_batch
        self.d_epoch = d_epoch
        self.keep = d_keep
        self.early = early

    def get_images(self):
        csv_path=input('Specify Path to Keyword CSV(ex. /path/to/file.csv): ')
        save_path = input('Specify Path to save category images(ex. /path/to/folder/): ')
        cat_im = input('How many images would you like to find for each keyword?: ')

        return csv_path, save_path, cat_im

    def get_nndata(self):
        img_h = input("Image Resize Height: ")
        img_w = input("Image Resize Width: ")
        val_sp = input("Specify % of data for validation(.00): ")

        return img_h, img_w, val_sp
    def get_defaults(self):
        print("Default Hyper parameters are:")
        print("Learning Rate: {}".format(self.dlr) +
              "\nBatch Size: {}".format(self.dbatch) +
              "\nEpochs: {}".format(self.d_epoch) +
              "\nDropOut Prob: {}".format(self.keep) +
              "\nEarl Stop: {}".format(self.early))
        option = input("\nPress(0) to continue or (1) for advanced settings")
        return int(option)

    def get_sess(self, advanced):

        if bool(advanced) == True:
            lr = input("learning rate: ")
            batch = input("batch size: ")
            epoch = input("How many epochs? ")
            keep_p = input("Specify keep probability if you would like to use drop out((1) for none): ")
            early_s = input('Use early stop(yes(1) no(0))? ')
        else:
            lr = self.dlr
            batch = self.dbatch
            epoch = self.d_epoch
            keep_p = self.keep
            early_s = self.early


        return lr, batch, epoch,keep_p,early_s



