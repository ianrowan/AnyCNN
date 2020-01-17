import GenBackend as gb
from GenUI import UI

def main():
    print("Welcome to GenNet")
    print("This program will guide you from data creation to Neural Network training in a few clicks\n")

    data = input('Select an option: \n (1) I have a data set \n (2) build me a data set')
    ui = UI(1e-4, 32, 8000, 1.0, 0)
    if data =='1':
        img_save = input("Specify path to dataset(/path/to/folder/: ")
    else:
        csv, img_save, num_imgs = ui.get_images()

        print("Scanning the web for 1000s of images.....\n")

        gb.GetImages().all_images(cat_path=csv, num_imgs=num_imgs, save_path=img_save)

        print("Download complete! Images may be found in {}".format(img_save))
        print("Please manually check accuracy of images(Our robots are learning how"
              "to do this as we speak!)")

    print("Lets build a Neural Network!" + "\n First we need your pre-processing preferences.....\n")

    im_h, im_w, val_sp = ui.get_nndata()

    net = gb.NeuralNet(int(im_h), int(im_w), float(val_sp), img_save)

    print("Class is in Session, lets train! \n")

    lr, batch, epoch, keep_prob, early_stop = ui.get_sess(ui.get_defaults())

    input("Press enter to commence training...")

    net.run_session(float(lr), int(batch), int(epoch), float(keep_prob), bool(int(early_stop)))

    print("Congratulations, you have just taught a computer to see!")


if __name__ == "__main__":
    main()
