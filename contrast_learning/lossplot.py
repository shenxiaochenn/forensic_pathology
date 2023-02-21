import os
class LossHistory():
    def __init__(self, log_dir, val_loss_flag=True,test_loss_flag=True,test2_loss_flag=False):
        import datetime
        self.time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.save_path = os.path.join(log_dir, "loss_" + str(self.time_str))
        self.val_loss_flag = val_loss_flag
        self.test_loss_flag = test_loss_flag
        self.test2_loss_flag = test2_loss_flag

        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []
        if self.test_loss_flag:
            self.test_loss = []
        if self.test2_loss_flag:
            self.test2_loss = []

        os.makedirs(self.save_path)

    def append_loss(self, loss=None, val_loss=None, test_loss=None,test2_loss=None):
        self.losses.append(loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")

        if self.val_loss_flag:
            self.val_loss.append(val_loss)
            with open(os.path.join(self.save_path, "epoch_loss_one_loss_" + str(self.time_str) + ".txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
        if self.test_loss_flag:
            self.test_loss.append(test_loss)
            with open(os.path.join(self.save_path, "epoch_loss_two_loss_" + str(self.time_str) + ".txt"), 'a') as f:
                f.write(str(test_loss))
                f.write("\n")
        if self.test2_loss_flag:
            self.test2_loss.append(test2_loss)
            with open(os.path.join(self.save_path, "epoch_loss_three_loss_" + str(self.time_str) + ".txt"), 'a') as f:
                f.write(str(test2_loss))
                f.write("\n")
